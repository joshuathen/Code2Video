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
            "Start with n points on a 2D plane.",
            "A line passes through one point, acting as pivot.",
            "The line rotates, sweeping like a radar.",
            "It hits another point and pivots again.",
            "We want to prove it hits every point."
        ]
        self.setup_layout("Problem Introduction: The Spinning Line", lecture_lines)
        
        # Load background radar icon
        radar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/radar.svg")
        self.place_in_area(radar, 'A1', 'F6', scale_factor=1.2)
        radar.set_opacity(0.3)
        self.add(radar)
        
        # 5 points
        points = VGroup(*[Dot(color=WHITE) for _ in range(5)])
        # Fix 21: Use A1-F6
        self.place_in_area(points, 'A1', 'F6', scale_factor=0.9)
        
        # Initial line (centered on one point)
        pivot_point = points[0].get_center()
        line = Line(start=pivot_point + LEFT * 2, end=pivot_point + RIGHT * 2, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(radar), FadeIn(points), Create(line))
        self.lecture[0].set_color(GRAY)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(YELLOW)
        # Rotation
        self.play(Rotate(line, angle=PI/4, about_point=pivot_point))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color(YELLOW)
        # Flash
        self.play(Flash(line, color=GREEN, line_length=0.2, flash_radius=0.3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color(YELLOW)
        self.play(Flash(radar, color=WHITE))
        self.wait(1)
