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
        self.setup_layout("Prerequisite: Angular Sweeping", ["We track the line's rotation.", "Sort points by their polar angle.", "The closest point becomes the pivot."])
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'B4', scale_factor=0.5)
        
        pivot = Dot(self.grid["C4"], color=WHITE)
        points = [
            Dot(self.grid["B3"], color=GREY),
            Dot(self.grid["B5"], color=GREY),
            Dot(self.grid["D5"], color=GREY),
            Dot(self.grid["E3"], color=GREY),
            Dot(self.grid["D1"], color=GREY),
            Dot(self.grid["B2"], color=GREY),
        ]
        
        sweep_line = Line(start=pivot.get_center(), end=pivot.get_center() + UP*1.5, color=YELLOW)
        
        animation_group = VGroup(compass, pivot, *points, sweep_line)
        self.place_in_area(animation_group, 'C3', 'F6', scale_factor=0.6)
        
        self.add(compass, pivot, *points, sweep_line)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Rotate(sweep_line, angle=2*PI, about_point=pivot.get_center(), rate_func=linear, run_time=3))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        # Simulating sorting by polar angle change
        for p in points:
             self.play(p.animate.set_color("#00FFFF"), run_time=0.3)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        self.play(pivot.animate.set_color(RED), run_time=1)
        self.wait(1)
