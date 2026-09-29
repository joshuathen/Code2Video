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
        lecture_lines = ["Now, map these moves to two dimensions.", "The path traces a Sierpinski triangle.", "Each disk expansion grows the fractal.", "Anchor points guide the pattern's evolution.", "The entire graph reveals the geometry."]
        self.setup_layout("Mapping the Sierpinski Triangle", lecture_lines)

        # Define colors
        color_main = "#457B9D"
        color_fill = "#1D3557"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        triangle = Polygon(UP * 2, LEFT * 2 + DOWN * 1.5, RIGHT * 2 + DOWN * 1.5, color=color_main)
        # Fix crowding issue: use place_in_area
        self.place_in_area(triangle, 'B2', 'D4', scale_factor=0.5)
        
        # Add asset
        disk = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        disk.move_to(triangle.get_center())
        self.play(Create(triangle), FadeIn(disk))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        # Fill inner triangle
        inner = Triangle(color=color_fill, fill_opacity=1.0)
        self.place_in_area(inner, 'B2', 'D4', scale_factor=0.25)
        self.play(FadeIn(inner))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Anchor points
        dots = VGroup(*[Dot(color=RED) for _ in range(3)])
        # Apply positioning fixes
        self.place_at_grid(dots[0], 'B3', scale_factor=0.8) # Line 85
        self.place_at_grid(dots[1], 'D2', scale_factor=0.8) # Line 86
        self.place_at_grid(dots[2], 'D4', scale_factor=0.8) # Line 87
        
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots]))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
