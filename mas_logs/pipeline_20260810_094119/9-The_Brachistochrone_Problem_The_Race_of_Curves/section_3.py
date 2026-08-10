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
        self.setup_layout("The Geometry of the Solution: The Cycloid", [
            "The solution is a special geometric shape.",
            "Meet the cycloid, traced by a rolling circle.",
            "Imagine a point on a spinning bike wheel.",
            "[Asset: cycloid_construction_animation]",
            "The cycloid defines our winning curve."
        ])
        
        # Assets
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg", color=WHITE)
        bicycle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bicycle.svg", color=WHITE)

        # === Animation for Lecture Line 1 ===
        # Draw a circle rolling to trace a path using wheel.svg
        self.place_at_grid(wheel, 'B4', scale_factor=0.6)
        self.add(wheel)
        
        # === Animation for Lecture Line 2 ===
        # Label the cycloid path as the solution #00FF00
        cycloid_label = Text("Cycloid Path", color=GREEN, font_size=20)
        self.place_in_area(cycloid_label, 'D4', 'F6', scale_factor=0.6)
        self.add(cycloid_label)
        
        # === Animation for Lecture Line 3 ===
        # Compare the cycloid with straight path #FF0000
        self.place_at_grid(bicycle, 'C2', scale_factor=0.6)
        self.add(bicycle)
        straight_path = Line(start=self.grid['C2'], end=self.grid['C5'], color=RED)
        self.add(straight_path)
        
        # === Animation for Lecture Line 5 ===
        # Highlight the minimum time path #FFFF00
        highlight = Dot(color=YELLOW)
        self.place_at_grid(highlight, 'C4', scale_factor=0.7)
        self.play(FadeIn(highlight))
