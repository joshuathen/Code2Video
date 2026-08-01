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
        # Setup Title and Lecture Lines
        title_text = "The Hidden Invariant: Counting Sides"
        lecture_lines = [
            "With seven points, one pivot leaves six others.",
            "Three stars stay on each side of the laser.",
            "As we pivot, the balance of points holds.",
            "The old pivot enters the opposite side’s set.",
            "Exactly three points always remain on each side."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        BLUE_PT = "#0000FF"
        RED_PT = "#FF0000"
        PIVOT_COLOR = "#FFFF00"
        LINE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # "With seven points, one pivot leaves six others."
        self.lecture[0].set_color(YELLOW)
        
        # Center of the windmill at D4 (Grid Anchor)
        pivot_pos = self.grid["D4"]
        pivot_dot = Dot(pivot_pos, color=PIVOT_COLOR, radius=0.12)
        
        # 3 Blue points (Left side) 
        blue_dots = VGroup(
            Dot(self.grid["C2"], color=BLUE_PT, radius=0.1),
            Dot(self.grid["D2"], color=BLUE_PT, radius=0.1),
            Dot(self.grid["E2"], color=BLUE_PT, radius=0.1)
        )
        
        # 3 Red points (Right side)
        red_dots = VGroup(
            Dot(self.grid["C6"], color=RED_PT, radius=0.1),
            Dot(self.grid["D6"], color=RED_PT, radius=0.1),
            Dot(self.grid["E6"], color=RED_PT, radius=0.1)
        )
        
        # Rotating line setup
        angle_tracker = ValueTracker(0)
        
        def get_line_points():
            angle = angle_tracker.get_value()
            direction = np.array([np.sin(angle), np.cos(angle), 0])
            center = pivot_dot.get_center()
            return [center - direction * 3.5, center + direction * 3.5]

        line = Line(*get_line_points(), color=LINE_COLOR, stroke_width=2)
        line.add_updater(lambda l: l.set_points_as_corners(get_line_points()))

        self.play(
            FadeIn(pivot_dot), 
            FadeIn(blue_dots), 
            FadeIn(red_dots), 
            Create(line)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Three stars stay on each side of the laser."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Count Labels (Issue 35 & 36 fix)
        left_count = Text("3", font_size=24, color=BLUE_PT)
        right_count = Text("3", font_size=24, color=RED_PT)
        self.place_at_grid(left_count, 'A3', scale_factor=1.2) # Resolved Issue 35
        self.place_at_grid(right_count, 'A5', scale_factor=1.2) # Resolved Issue 36
        
        self.play(Write(left_count), Write(right_count))
        self.play(Indicate(blue_dots, color=BLUE_PT), Indicate(red_dots, color=RED_PT))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "As we pivot, the balance of points holds."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Calculate angle to hit the first red point (C6)
        target_pos = self.grid["C6"]
        rel_vec = target_pos - pivot_pos
        # Note: arctan2(x, y) gives angle from Y-axis clockwise
        target_angle = np.arctan2(rel_vec[0], rel_vec[1])
        
        # Rotate to the hit
        self.play(angle_tracker.animate.set_value(target_angle), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # "The old pivot enters the opposite side’s set."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Transitioning the pivot
        old_pivot_center = pivot_dot.get_center()
        new_pivot_ref = red_dots[0]
        
        # New red dot representing the old pivot joining the side set
        old_pivot_as_side = Dot(old_pivot_center, color=RED_PT, radius=0.1)
        red_dots.add(old_pivot_as_side)
        
        self.play(
            new_pivot_ref.animate.set_color(PIVOT_COLOR),
            FadeIn(old_pivot_as_side),
            Flash(new_pivot_ref, color=YELLOW)
        )
        
        # Update logical pivot position
        line.clear_updaters()
        pivot_dot.move_to(target_pos)
        new_pivot_ref.set_opacity(0) # Conceal static member since pivot_dot covers it
        line.add_updater(lambda l: l.set_points_as_corners(get_line_points()))
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Exactly three points always remain on each side."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Continue rotation slightly to show the invariant in motion
        self.play(angle_tracker.animate.increment_value(0.6), run_time=1.5, rate_func=linear)
        self.play(Indicate(left_count), Indicate(right_count))
        
        self.wait(2)
