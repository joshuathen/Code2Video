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
        # Section data
        title_text = "Introduction: The Infinite Spinner"
        lecture_lines = [
            "Imagine points like stars scattered across a map.",
            "We place a rotating line through one central point.",
            "As the line spins, it strikes another point.",
            "The pivot point then jumps to this new star.",
            "Can this simple dance visit every single point?"
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Define point positions on the grid (A2 to E6 as per Issue 21)
        # Avoid Column 1 (x=0.5) to prevent overlap with lecture
        p1_pos = self.grid['C3'] # Central point (Initial Pivot)
        p2_pos = self.grid['B2']
        p3_pos = self.grid['D5']
        p4_pos = self.grid['F4']
        p5_pos = self.grid['B6']
        
        points = VGroup(
            Dot(p1_pos, color="#FFFFFF"),
            Dot(p2_pos, color="#FFFFFF"),
            Dot(p3_pos, color="#FFFFFF"),
            Dot(p4_pos, color="#FFFFFF"),
            Dot(p5_pos, color="#FFFFFF")
        )

        # Issue 22: Ensure central point is anchored properly
        self.place_at_grid(points[0], 'C3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Fade in 5 points at random positions (color: #FFFFFF). self.wait(1.5).
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            FadeIn(points)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Create a long line through a central point (color: #FFD700). self.wait(2.0).
        angle_tracker = ValueTracker(0)
        # We'll use a local copy for the pivot coordinate that we can update
        current_pivot_coord = p1_pos.copy()
        
        # Issue 20: Keep line within grid bounds (A1-F6, scale 0.9)
        # Grid width is 5. Max line half-length 2.25.
        line_half_length = 2.25
        
        windmill_line = Line(
            current_pivot_coord - np.array([1, 0, 0]) * line_half_length,
            current_pivot_coord + np.array([1, 0, 0]) * line_half_length,
            color="#FFD700"
        )
        
        def windmill_updater(mob):
            angle = angle_tracker.get_value()
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            start = current_pivot_coord - direction * line_half_length
            end = current_pivot_coord + direction * line_half_length
            mob.set_points_as_corners([start, end])
            
        windmill_line.add_updater(windmill_updater)
        
        self.play(
            self.lecture[1].animate.set_color("#FFD700"),
            Create(windmill_line)
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Rotate the line clockwise; when it hits another point, flash that point (color: #00FF00). self.wait(1.5).
        # Vector P1->P3 (C3 to D5) is (2.0, -1.0). Angle is atan2(-1, 2)
        target_angle_1 = np.arctan2(-1.0, 2.0) # Clockwise from 0
        
        self.play(
            self.lecture[2].animate.set_color("#00FF00"),
            angle_tracker.animate.set_value(target_angle_1),
            run_time=2.5,
            rate_func=rate_functions.linear
        )
        # Flash the hit point (P3)
        self.play(Indicate(points[2], color="#00FF00"))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # Shift the line's pivot point to the hit point and continue rotation. self.wait(2.0).
        # Update current_pivot_coord in-place
        current_pivot_coord[:] = p3_pos
        
        # Next target: P4 (F4). Vector P3->P4 is (-1.0, -2.0). Angle is atan2(-2, -1)
        target_angle_2 = np.arctan2(-2.0, -1.0)
        # Ensure clockwise (decreasing)
        while target_angle_2 > target_angle_1:
            target_angle_2 -= TAU
            
        self.play(
            self.lecture[3].animate.set_color("#FFD700"),
            angle_tracker.animate.set_value(target_angle_2),
            run_time=3.0,
            rate_func=rate_functions.linear
        )
        # Flash the new hit point (P4)
        self.play(Indicate(points[3], color="#00FF00"))
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # Highlight all points with a pulsing glow to indicate potential visitation (color: #FF4500). self.wait(2.0).
        self.play(
            self.lecture[4].animate.set_color("#FF4500"),
            *[Indicate(p, color="#FF4500") for p in points]
        )
        self.wait(2.0)
