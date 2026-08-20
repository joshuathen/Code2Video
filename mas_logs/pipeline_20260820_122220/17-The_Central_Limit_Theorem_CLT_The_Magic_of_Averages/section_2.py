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
            "Real-world populations are often skewed and chaotic.",
            "Yet, averages of samples behave very predictably.",
            "Skewed data transforms into a perfect bell curve."
        ]
        self.setup_layout("The Problem: Chaos vs. Order", lecture_lines)
        
        # Assets
        pop_svg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg"
        
        # === Animation for Lecture Line 1 ===
        # Use SVGMobject for population icon as requested
        dots = VGroup(*[SVGMobject(pop_svg_path, color="#FF0000", height=0.2) for _ in range(30)])
        for dot in dots:
            # Random scatter within an area
            dot.shift(np.random.uniform(-1.0, 1.0, 3) * 0.5)
        
        # Fix 1: Place in area B1 to E4
        self.place_in_area(dots, "B1", "E4", scale_factor=0.9)
        self.play(FadeIn(dots))
        self.lecture[0].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create an unstable trend line
        trend = Line(start=self.grid["B2"], end=self.grid["E5"], color="#FFA500")
        self.play(Create(trend))
        self.lecture[1].set_color("#FFA500")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fix 2: Highlight circle placement
        highlight = Circle(radius=0.5, color="#FFFFFF", stroke_width=2)
        self.place_in_area(highlight, "C2", "D3", scale_factor=0.85)
        
        # Fix 3: Adjust dots placement (re-using existing dots group as per instructions)
        # Note: self.place_in_area will rescale relative to original, so use 1/0.9 to reset
        dots.scale(1/0.9) 
        self.place_in_area(dots, "B1", "E3", scale_factor=0.8)
        
        self.play(Create(highlight), FadeIn(dots))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
