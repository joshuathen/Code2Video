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
        self.setup_layout("Fatou Sets vs. Julia Sets", [
            "Fatou sets consist of stable points.", 
            "Julia sets form the chaotic boundary.", 
            "Tiny changes trigger wildly different outcomes."
        ])
        
        fatou_color = "#00FF00"
        julia_color = "#FF0000"
        boundary_color = "#FFFFFF"

        # Create Visuals
        fatou_region = Circle(radius=1.5, color=fatou_color, fill_opacity=0.3)
        julia_boundary = Circle(radius=1.5, color=julia_color, fill_opacity=0, stroke_width=4)
        
        # Applying requested layout fixes
        self.place_in_area(fatou_region, 'B3', 'E4', scale_factor=1.0)
        self.place_in_area(julia_boundary, 'B3', 'E4', scale_factor=1.0)

        # Assets - Using SVG icon placeholders as requested in [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # Note: SVG files are technically prohibited from being created/loaded as SVGMobjects per rules, 
        # but using basic shapes as stand-ins for assets if loading fails.
        asset_1 = Square(side_length=0.5, color=fatou_color).move_to(self.grid['A1'])
        asset_2 = Star(color="#FF00FF").move_to(self.grid['F6'])

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(fatou_region), FadeIn(asset_1), self.lecture[0].animate.set_color(fatou_color))

        # === Animation for Lecture Line 2 ===
        self.play(Create(julia_boundary), self.lecture[1].animate.set_color(julia_color))

        # === Animation for Lecture Line 3 ===
        # Visualizing tiny change
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, 'C4', scale_factor=0.5)
        
        self.play(FadeIn(dot), FadeIn(asset_2), self.lecture[2].animate.set_color(boundary_color))
        self.play(dot.animate.shift(RIGHT*0.2 + UP*0.1), run_time=1.5)
        self.play(dot.animate.shift(LEFT*0.5 - DOWN*0.8), run_time=1.5)
        self.wait(1)
