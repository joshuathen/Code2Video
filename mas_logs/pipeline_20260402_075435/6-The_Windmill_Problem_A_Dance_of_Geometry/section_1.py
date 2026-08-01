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
        # Section Title and Lecture Lines
        title_text = "Introduction: Pip and the Star Field"
        lecture_lines = [
            'Imagine a star field with seven glowing points.',
            'A robot, Pip, starts at pivot star A.',
            'Pip holds a long, rotating laser beam.',
            'The laser spins clockwise around the pivot point.',
            'When it hits star B, the pivot will shift.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Imagine a star field with seven glowing points.
        self.lecture[0].set_color(WHITE)
        
        # Create a field of 7 white points (#FFFFFF) representing stars.
        star_field = VGroup(*[Dot(radius=0.08, color="#FFFFFF") for _ in range(7)])
        # Issue 27 fix: use place_in_area to avoid obstruction of lecture notes.
        self.place_in_area(star_field, 'A2', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(star_field))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A robot, Pip, starts at pivot star A.
        self.lecture[1].set_color("#FFFF00") # Yellow highlight
        
        star_A = star_field[0]
        # Issue 28 fix: anchor pivot star A and label to the grid.
        self.place_at_grid(star_A, 'C3', scale_factor=1.0)
        self.play(star_A.animate.set_color("#FFFF00"))
        
        label_A = Text("A", color="#FFFF00", font_size=24)
        self.place_at_grid(label_A, 'D3', scale_factor=0.7)
        self.play(Write(label_A))

        # === Animation for Lecture Line 3 ===
        # Pip holds a long, rotating laser beam.
        self.lecture[2].set_color("#FF0000") # Red highlight
        
        pivot_pos = star_A.get_center()
        angle_tracker = ValueTracker(45 * DEGREES)
        
        init_vec = np.array([np.cos(45*DEGREES), np.sin(45*DEGREES), 0])
        laser = Line(
            pivot_pos - 5 * init_vec,
            pivot_pos + 5 * init_vec,
            color="#FF0000",
            stroke_width=3
        )
        
        def laser_updater(m):
            ang = angle_tracker.get_value()
            vec = np.array([np.cos(ang), np.sin(ang), 0])
            m.set_points_by_ends(pivot_pos - 5 * vec, pivot_pos + 5 * vec)
        
        laser.add_updater(laser_updater)
        self.play(Create(laser))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # The laser spins clockwise around the pivot point.
        self.lecture[3].set_color("#FF0000") # Red color for consistency with laser
        
        # Target star B positioning
        star_B = star_field[1]
        # Issue 29 fix: anchor target star B to the grid.
        self.place_at_grid(star_B, 'B5', scale_factor=1.0)
        
        pos_B = star_B.get_center()
        target_angle = np.arctan2(pos_B[1] - pivot_pos[1], pos_B[0] - pivot_pos[0])
        
        # Rotation clockwise (angle tracker decrease from 45 deg to target angle)
        self.play(
            angle_tracker.animate.set_value(target_angle),
            run_time=3,
            rate_func=linear
        )
        self.wait(0.2)

        # === Animation for Lecture Line 5 ===
        # When it hits star B, the pivot will shift.
        self.lecture[4].set_color("#00FFFF") # Cyan highlight
        
        label_B = Text("B", color="#00FFFF", font_size=24)
        # Issue 29 fix: anchor label B to the grid.
        self.place_at_grid(label_B, 'A5', scale_factor=0.7)

        # Visual feedback: flash B cyan and reveal label
        self.play(star_B.animate.set_color("#00FFFF"), FadeIn(label_B), run_time=0.5)
        self.play(star_B.animate.set_color("#FFFFFF"), run_time=0.2)
        self.play(star_B.animate.set_color("#00FFFF"), run_time=0.2)
        
        self.wait(2)
