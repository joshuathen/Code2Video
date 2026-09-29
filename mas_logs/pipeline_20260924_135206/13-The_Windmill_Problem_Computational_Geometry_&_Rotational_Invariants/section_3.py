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
        lecture_lines = [
            "Geometry defines how the line pivots.",
            "The line snaps to the next available point.",
            "Think of it like a shifting bicycle gear.",
            "This continues for every point encounter.",
            "Geometry ensures a smooth, continuous transition."
        ]
        self.setup_layout("The Core Mechanism: The Pivot Rule", lecture_lines)
        
        # Load assets
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg", color="#00FF00")
        bicycle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bicycle.svg", color="#FFFFFF")
        
        pivot_group = VGroup(gear)
        
        # Applying requested layout adjustments
        self.place_in_area(pivot_group, 'A4', 'B6', scale_factor=0.5)
        self.place_at_grid(bicycle, 'D4', scale_factor=0.4)
        
        rotating_line = Line(start=pivot_group.get_center(), end=pivot_group.get_center() + RIGHT*1.5, color="#FFFF00")
        
        path = Arc(radius=1.5, start_angle=0, angle=PI/2, color="#FFFFFF")
        path.move_to(pivot_group.get_center())
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(pivot_group))
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        self.play(Create(rotating_line))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(bicycle))
        self.lecture[2].set_color("#FFFFFF")

        # === Animation for Lecture Line 4 ===
        self.play(Create(path), run_time=1)
        self.play(Rotate(rotating_line, angle=PI/2, about_point=pivot_group.get_center(), run_time=2))
        self.lecture[3].set_color("#00FF00")

        # === Animation for Lecture Line 5 ===
        self.play(Indicate(rotating_line, color=WHITE))
        self.lecture[4].set_color("#FFFF00")
        
        self.wait(2)
