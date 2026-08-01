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
        # Setup the layout with the lecture lines for section 4
        self.setup_layout(
            "The Core Invariant: Counting Sides",
            [
                "Consider the points on each side of the line.",
                "Color points on the left red and right blue.",
                "As the pivot swaps, a point moves through the pivot.",
                "Crucially, the number of points on each side stays constant.",
                "This invariant is the key to solving the problem."
            ]
        )

        # Colors for points, labels, and the rotating line
        COLOR_LEFT_LABEL = "#00FFFF"  # Cyan label for the left side
        COLOR_RIGHT_LABEL = "#FF00FF" # Magenta label for the right side
        COLOR_RED = "#FF0000"        # Left-side points
        COLOR_BLUE = "#0000FF"       # Right-side points
        COLOR_LINE = "#FFFFFF"       # The windmill line
        COLOR_PIVOT = "#FFFFFF"      # Pivot point color

        # Grid positions for points and labels (Addressing VideoCritic Issues 35, 36, 37)
        # Pivot initially at C3
        pivot_pos = self.grid["C3"]
        
        # 4 points on the left (Column 1)
        left_pos_keys = ["A1", "B1", "D1", "E1"]
        # 4 points on the right (Column 5)
        right_pos_keys = ["A5", "B5", "D5", "E5"]
        
        # Create point mobjects
        pivot_dot = Dot(pivot_pos, color=COLOR_PIVOT, radius=0.1)
        left_dots = VGroup(*[Dot(self.grid[p], color=WHITE, radius=0.08) for p in left_pos_keys])
        right_dots = VGroup(*[Dot(self.grid[p], color=WHITE, radius=0.08) for p in right_pos_keys])
        
        # Create count labels "4" for each side (Relocated to Row B as per Issue 37)
        left_label = Text("4", font_size=36, color=COLOR_LEFT_LABEL)
        self.place_at_grid(left_label, "B2", scale_factor=0.8) # Issue 36: Near left points, safe from margin
        
        right_label = Text("4", font_size=36, color=COLOR_RIGHT_LABEL)
        self.place_at_grid(right_label, "B6", scale_factor=0.8) # Issue 35: Clear of blue points

        # Rotating line setup using ValueTracker
        line_angle = ValueTracker(PI/2) # Start vertical
        self.current_pivot_pos = pivot_pos
        
        # Create the line mobject
        rotating_line = Line(ORIGIN, UP, color=COLOR_LINE, stroke_width=2)
        
        # Define the updater for the line
        def line_updater(l):
            angle = line_angle.get_value()
            p = self.current_pivot_pos
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            # Extend symmetrically from the pivot
            l.put_start_and_end_on(p - direction * 3.5, p + direction * 3.5)
            
        rotating_line.add_updater(line_updater)

        # === Animation for Lecture Line 1 ===
        # Consider the points on each side of the line.
        self.lecture[0].set_color(YELLOW)
        self.add(pivot_dot, left_dots, right_dots)
        self.play(Create(rotating_line))
        self.wait(1.0)

        # === Animation for Lecture Line 2 ===
        # Color points on the left red and right blue.
        self.lecture[1].set_color(YELLOW)
        self.play(
            left_dots.animate.set_color(COLOR_RED),
            right_dots.animate.set_color(COLOR_BLUE),
            Write(left_label),
            Write(right_label)
        )
        self.wait(1.0)

        # === Animation for Lecture Line 3 ===
        # As the pivot swaps, a point moves through the pivot.
        self.lecture[2].set_color(YELLOW)
        
        # Target point for swap: B5 (Column 5, Row B)
        # Vector from C3(2.5, 0.2) to B5(4.5, 1.2) is (2, 1). Angle is atan2(1, 2)
        target_angle_val = np.arctan2(1, 2)
        
        # Rotate clockwise (decreasing angle from PI/2)
        self.play(line_angle.animate.set_value(target_angle_val), run_time=2.0, rate_func=linear)
        
        # Perform the pivot swap:
        # 1. B5 (right_dots[1]) becomes the new white pivot.
        # 2. C3 (old pivot) moves to the "right" side relative to rotation and becomes blue.
        # This keeps the total counts (4 Red, 4 Blue) invariant.
        self.play(
            right_dots[1].animate.set_color(COLOR_PIVOT).set_radius(0.1),
            pivot_dot.animate.set_color(COLOR_BLUE).set_radius(0.08),
            run_time=0.2
        )
        # Update the pivot position for the line updater
        self.current_pivot_pos = self.grid["B5"]
        
        # Continue the rotation slightly to show the line passing through
        self.play(line_angle.animate.set_value(target_angle_val - 0.3), run_time=1.0, rate_func=linear)

        # === Animation for Lecture Line 4 ===
        # Crucially, the number of points on each side stays constant.
        self.lecture[3].set_color(YELLOW)
        self.play(
            Flash(left_label, color=COLOR_LEFT_LABEL, flash_radius=0.4),
            Flash(right_label, color=COLOR_RIGHT_LABEL, flash_radius=0.4),
            left_label.animate.scale(1.2),
            right_label.animate.scale(1.2)
        )
        self.play(
            left_label.animate.scale(1.0/1.2),
            right_label.animate.scale(1.0/1.2)
        )
        self.wait(1.0)

        # === Animation for Lecture Line 5 ===
        # This invariant is the key to solving the problem.
        self.lecture[4].set_color(YELLOW)
        # Final rotation to emphasize movement
        self.play(line_angle.animate.set_value(target_angle_val - 1.0), run_time=2.0)
        self.wait(2.0)
