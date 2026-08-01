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
            "- Start with a set of finite points in space.",
            "- A line passes through one point, called the pivot.",
            "- The line rotates clockwise around this pivot point.",
            "- It continues until it hits another point in the set.",
            "- The hit point becomes the new center of rotation."
        ]
        self.setup_layout("Introduction: The Dancing Line", lecture_lines)

        # Colors
        COLOR_POINTS = "#FFFFFF"
        COLOR_LINE = "#00FFFF"
        COLOR_HIT = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # - Start with a set of finite points in space.
        self.lecture[0].set_color(COLOR_POINTS)
        
        # Use grid alignment for individual points (Issue 28)
        point_coords = ["B2", "B4", "C3", "C5", "D2", "D4", "D6", "E3", "E5", "F4"]
        points = VGroup(*[Dot(radius=0.08, color=COLOR_POINTS) for _ in range(10)])
        for dot, coord in zip(points, point_coords):
            self.place_at_grid(dot, coord)
        
        # Center the group in the right area to avoid lecture notes (Issue 27)
        points_group = points
        self.place_in_area(points_group, 'B2', 'F6', scale_factor=0.7)
        
        self.play(FadeIn(points_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # - A line passes through one point, called the pivot.
        self.lecture[1].set_color(COLOR_LINE)
        
        # Pivot point (e.g. the one originally at C3, now moved by group placement)
        pivot_point = points[2]
        
        # Create dancing line (Issue 29 - kept shorter to avoid clutter)
        dancing_line = Line(start=LEFT*1.8, end=RIGHT*1.8, color=COLOR_LINE)
        dancing_line.move_to(pivot_point.get_center())
        dancing_line.set_angle(120 * DEGREES)
        
        self.play(Create(dancing_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # - The line rotates clockwise around this pivot point.
        self.lecture[2].set_color(COLOR_LINE)
        
        # Hit another point (e.g. the one originally at D4)
        hit_point = points[5]
        
        # Rotation logic
        v = hit_point.get_center() - pivot_point.get_center()
        target_angle = np.arctan2(v[1], v[0])
        current_angle = dancing_line.get_angle()
        
        rot_angle = target_angle - current_angle
        while rot_angle > 0:
            rot_angle -= 2 * PI
            
        self.play(
            Rotate(dancing_line, angle=rot_angle, about_point=pivot_point.get_center()),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # - It continues until it hits another point in the set.
        self.lecture[3].set_color(COLOR_HIT)
        
        self.play(
            pivot_point.animate.set_color(COLOR_HIT).scale(1.5),
            hit_point.animate.set_color(COLOR_HIT).scale(1.5),
            dancing_line.animate.set_color(COLOR_HIT)
        )
        self.play(
            Flash(pivot_point, color=COLOR_HIT),
            Flash(hit_point, color=COLOR_HIT)
        )
        self.play(
            pivot_point.animate.scale(1/1.5),
            hit_point.animate.scale(1/1.5)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # - The hit point becomes the new center of rotation.
        self.lecture[4].set_color(COLOR_LINE)
        
        # Scale everything up to emphasize motion
        everything = VGroup(points_group, dancing_line)
        self.play(
            everything.animate.scale(1.2),
            dancing_line.animate.set_color(COLOR_LINE)
        )
        self.wait(2)
