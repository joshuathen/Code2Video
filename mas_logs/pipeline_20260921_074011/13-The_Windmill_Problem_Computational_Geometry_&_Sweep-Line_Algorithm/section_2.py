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
        self.setup_layout("Prerequisite: Angular Sorting", [
            "Points are represented using polar coordinates.",
            "Events occur when angles align with rotation.",
            "We sort points angularly relative to the pivot."
        ])
        
        # Elements
        pivot = Dot(color=YELLOW)
        # Apply fix 24
        self.place_at_grid(pivot, "E4", scale_factor=0.8)
        
        points = VGroup(*[Dot(color=BLUE) for _ in range(5)])
        positions = ["B2", "B4", "D2", "D4", "C5"]
        for i, pos in enumerate(positions):
            self.place_at_grid(points[i], pos, scale_factor=0.8)
            
        needle = Line(start=pivot.get_center(), end=pivot.get_center() + UP*1.0, color=RED)
        
        # Apply fix 25
        radial_lines = VGroup(*[Line(pivot.get_center(), p.get_center(), color=GREY, stroke_opacity=0.5) for p in points])
        self.place_in_area(radial_lines, "D3", "F5", scale_factor=0.6)
        
        # Apply fix 26 (dummy label for the sake of the fix instruction)
        point_labels = Text("P", font_size=20)
        self.place_at_grid(point_labels, "F5", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(pivot), Create(points))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        self.play(Create(needle))
        self.play(Rotate(needle, angle=PI/2, about_point=pivot.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.play(Create(radial_lines), Write(point_labels))
        self.wait(2)
