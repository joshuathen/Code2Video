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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Move to the complex plane.", "Assign colors by the root found.", "Boundaries reveal infinitely complex patterns."]
        self.setup_layout("Visualizing Chaos: Newton Fractals", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Initialize grid
        plane_area = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        plane_area.set_color("#FFFFFF")
        # Fix #30: plane area position
        self.place_in_area(plane_area, "B2", "F6", scale_factor=0.75)
        self.play(FadeIn(plane_area))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Coloring by root
        self.lecture[1].set_color("#0000FF")
        
        # Fix #19: pixels asset
        pixels = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixels.svg")
        pixels.set_color("#0000FF")
        self.place_at_grid(pixels, "C5", scale_factor=0.8)
        self.play(FadeIn(pixels))
        
        # === Animation for Lecture Line 3 ===
        # Fractal zoom
        self.lecture[2].set_color("#FF00FF")
        
        # Fix #29/31: fractal pattern placement
        fractal_pattern = VGroup(
            Square(color="#FF00FF", fill_opacity=0.5).scale(0.5),
            Square(color="#FFFF00", fill_opacity=0.5).scale(0.25).shift(UP*0.5),
            Square(color="#00FFFF", fill_opacity=0.5).scale(0.125).shift(DOWN*0.5)
        )
        self.place_at_grid(fractal_pattern, "D4", scale_factor=0.8)
        self.play(DrawBorderThenFill(fractal_pattern), run_time=2)
        self.wait(1)
