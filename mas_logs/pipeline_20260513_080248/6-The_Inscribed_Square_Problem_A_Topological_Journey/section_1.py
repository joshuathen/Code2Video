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
        # Setup layout
        lines = [
            'Imagine a square table on an uneven loop.',
            'Can all four legs touch the path simultaneously?',
            'This is the famous Inscribed Square Problem.',
            'The loop is a flexible Jordan curve.',
            'We seek four points forming square vertices.'
        ]
        self.setup_layout("Introduction: The Table on the Uneven Floor", lines)

        # Colors
        HIGHLIGHT_COLOR = "#FFFF00"
        CURVE_COLOR = "#FFFFFF"
        VERTEX_COLOR = "#FF0000"
        SQUARE_COLOR = "#4444FF"

        # === Animation for Lecture Line 1 ===
        # Imagine a square table on an uneven loop.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create an irregular loop (Jordan Curve)
        curve_points = [
            [1, 1, 0], [2, 1.5, 0], [3, 1, 0], [3.5, 0, 0],
            [3, -1, 0], [2, -1.8, 0], [0.5, -1.5, 0], [0, -0.5, 0],
            [0.5, 0.5, 0]
        ]
        loop = Polygon(*curve_points, color=CURVE_COLOR).set_stroke(width=3)
        loop.make_smooth()
        # Resolved Issue 32: Reduced scale and centered in B2-F5
        self.place_in_area(loop, "B2", "F5", scale_factor=1.0)
        
        # Create a square with red vertices
        square_frame = Square(side_length=1.5, color=SQUARE_COLOR, stroke_opacity=0.5)
        vertices = VGroup(*[
            Dot(square_frame.get_vertices()[i], color=VERTEX_COLOR, radius=0.08)
            for i in range(4)
        ])
        table = VGroup(square_frame, vertices)
        # Resolved Issue 33: Centered table in area C3-D4
        self.place_in_area(table, "C3", "D4", scale_factor=0.7)

        self.play(Create(loop), run_time=1.5)
        self.play(FadeIn(table))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Can all four legs touch the path simultaneously?
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)

        # Move and rotate the square showing it missing the curve
        self.play(
            table.animate.rotate(0.5).shift(RIGHT * 0.5 + UP * 0.3),
            run_time=1.5
        )
        self.play(
            table.animate.rotate(-0.8).shift(LEFT * 0.8 + DOWN * 0.5),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This is the famous Inscribed Square Problem.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        # More rotation to emphasize the "searching"
        self.play(
            table.animate.rotate(2.0 * PI / 3).scale(1.1),
            run_time=2,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The loop is a flexible Jordan curve.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)

        # Transform loop into a different irregular shape
        new_curve_points = [
            [0, 1.2, 0], [1.5, 1.8, 0], [3, 1.2, 0], [4, 0, 0],
            [3, -1.5, 0], [1.5, -0.5, 0], [0, -1.5, 0], [-1, 0, 0]
        ]
        loop_new = Polygon(*new_curve_points, color=CURVE_COLOR).set_stroke(width=3)
        loop_new.make_smooth()
        # Resolved Issue 34: Used consistent B2-F5 area and scale 1.0
        self.place_in_area(loop_new, "B2", "F5", scale_factor=1.0)
        
        self.play(Transform(loop, loop_new), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # We seek four points forming square vertices.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)

        # Align square so vertices "magically" touch the new curve
        # Move back to the loop's center area
        area_center_x = (self.grid["B2"][0] + self.grid["F5"][0]) / 2
        area_center_y = (self.grid["B2"][1] + self.grid["F5"][1]) / 2
        area_center = np.array([area_center_x, area_center_y, 0])
        target_rotation = PI/6
        
        self.play(
            table.animate.move_to(area_center).rotate(target_rotation).scale(0.9),
            run_time=2
        )

        # Pulse effect: highlight when all vertices "lie" on the curve
        pulse_square = square_frame.copy().set_color(YELLOW).set_stroke(width=8)
        self.play(
            Flash(table, color=VERTEX_COLOR, flash_radius=1.5),
            FadeIn(pulse_square),
            run_time=1
        )
        self.play(
            pulse_square.animate.scale(1.2).set_stroke(opacity=0),
            run_time=0.8,
            rate_func=slow_into
        )
        self.remove(pulse_square)
        
        self.wait(2)
