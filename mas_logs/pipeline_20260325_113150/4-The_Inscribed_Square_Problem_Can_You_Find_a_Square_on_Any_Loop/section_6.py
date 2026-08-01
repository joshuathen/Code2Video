from manim import *
import numpy as np
import os
from pathlib import Path

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout
        self.setup_layout("The Conclusion: The Intersection Proof", [
            "A Möbius strip must intersect itself in 3D.",
            "At the intersection, two distinct pairs overlap.",
            "They share a midpoint and have equal length.",
            "These four points form the corners of a rectangle.",
            "We have found our inscribed shape on the loop!"
        ])

        # Color constants
        strip_color = "#00BFFF"
        highlight_color = "#FFFF00"
        rectangle_color = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(strip_color)
        
        mobius_projection = ParametricFunction(
            lambda t: np.array([1.8 * np.sin(t), 1.2 * np.sin(2 * t), 0]),
            t_range=[0, TAU],
            color=strip_color
        )
        self.place_in_area(mobius_projection, "A3", "E5")
        
        self.play(Create(mobius_projection), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(highlight_color)
        
        intersection_dot = Dot(color=highlight_color)
        self.place_at_grid(intersection_dot, "C4")
        
        self.play(FadeIn(intersection_dot))
        self.play(
            intersection_dot.animate.scale(1.5),
            rate_func=there_and_back,
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(highlight_color)
        
        center_pt = self.grid["C4"]
        diag_offset_1 = np.array([1.2, 0.8, 0])
        diag_offset_2 = np.array([-1.2, 0.8, 0])
        
        segment1 = Line(center_pt - diag_offset_1, center_pt + diag_offset_1, color=highlight_color)
        segment2 = Line(center_pt - diag_offset_2, center_pt + diag_offset_2, color=highlight_color)
        
        self.play(
            Create(segment1),
            Create(segment2),
            intersection_dot.animate.set_opacity(0.3),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(rectangle_color)
        self.play(FadeOut(mobius_projection))

        loop = ParametricFunction(
            lambda t: np.array([
                2.2 * np.cos(t) + 0.3 * np.sin(3 * t),
                1.5 * np.sin(t) + 0.3 * np.cos(2 * t),
                0
            ]),
            t_range=[0, TAU],
            color=strip_color,
            stroke_opacity=0.4
        )
        self.place_in_area(loop, "A3", "E5")
        
        rect_points = [
            segment1.get_start(),
            segment2.get_start(),
            segment1.get_end(),
            segment2.get_end()
        ]
        ordered_rect_points = [rect_points[0], rect_points[1], rect_points[2], rect_points[3]]
        rectangle = Polygon(*ordered_rect_points, color=rectangle_color, stroke_width=5)
        
        self.play(
            Create(loop),
            Create(rectangle),
            segment1.animate.set_stroke(opacity=0.2),
            segment2.animate.set_stroke(opacity=0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(rectangle_color)
        
        corner_dots = VGroup(*[Dot(p, color=rectangle_color, radius=0.1) for p in ordered_rect_points])
        
        self.play(FadeIn(corner_dots))
        self.play(
            *[Flash(p, color=rectangle_color) for p in ordered_rect_points],
            run_time=1.5
        )
        
        self.play(
            rectangle.animate.set_stroke(width=10),
            corner_dots.animate.scale(1.5),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
