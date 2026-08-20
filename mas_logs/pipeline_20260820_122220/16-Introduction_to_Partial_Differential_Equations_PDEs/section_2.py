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
        lecture_lines = [
            "The Heat Equation models spatial diffusion.",
            "Heat flows from hot to cold regions.",
            "Concavity drives this thermal equalization process."
        ]
        self.setup_layout("The Heat Equation: Diffusion in Motion", lecture_lines)
        
        # Paths for Assets
        rod_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg"
        
        # === Animation for Lecture Line 1 ===
        # Use SVG for rod
        rod = SVGMobject(rod_path, color=WHITE)
        self.place_at_grid(rod, 'E2', scale_factor=0.7)
        self.play(Create(rod))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Use SVG for thermal distribution representation (using rod again as per instructions)
        temp_curve = SVGMobject(rod_path, color="#FF0000")
        self.place_in_area(temp_curve, 'A2', 'C5', scale_factor=0.8)
        self.play(Create(temp_curve))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        # Represent concavity with vectors
        arrows = VGroup(*[
            Arrow(start=self.grid[f'C{i}'], end=self.grid[f'D{i}'], color="#00FFFF", buff=0.1)
            for i in range(2, 6)
        ])
        # Add a placeholder for diffusion flow visualization if needed using the third asset requirement
        diffusion_vis = SVGMobject(rod_path, color="#00FFFF").scale(0.3)
        self.place_at_grid(diffusion_vis, 'D3')
        
        self.play(Create(arrows), FadeIn(diffusion_vis))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
