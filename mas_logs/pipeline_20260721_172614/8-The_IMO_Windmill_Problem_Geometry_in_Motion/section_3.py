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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup title and lines
        title_text = "The Magic Invariant: The Balancing Act"
        lecture_lines = [
            "The line divides the other points into two sets.",
            "Let's color points on each side differently to see.",
            "As the pivot switches, the side-counts remain perfectly balanced.",
            "This invariant property holds throughout the entire rotation.",
            "The number of points on each side never changes."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_LINE = "#FFFF00"
        COLOR_PIVOT = "#2ECC71"
        COLOR_LEFT = "#3498DB"
        COLOR_RIGHT = "#E74C3C"

        # Create points
        p_pivot = Dot(color=COLOR_PIVOT)
        p_l1 = Dot(color=WHITE)
        p_l2 = Dot(color=WHITE)
        p_r1 = Dot(color=WHITE)
        p_r2 = Dot(color=WHITE)

        # Position points on grid (Resolving Issues 30, 31, 32)
        # Shifted right to avoid lecture text and use space efficiently
        self.place_at_grid(p_pivot, 'D4')
        self.place_at_grid(p_l1, 'B3')
        self.place_at_grid(p_l2, 'E3')
        self.place_at_grid(p_r1, 'B5')
        self.place_at_grid(p_r2, 'E5')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_LINE)
        pivot_pos = p_pivot.get_center()
        # Initial vertical line (angle = PI/2)
        line = Line(pivot_pos + 4.0*UP, pivot_pos + 4.0*DOWN, color=COLOR_LINE, stroke_width=2)
        
        self.add(p_pivot, p_l1, p_l2, p_r1, p_r2)
        self.play(Create(line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_LEFT)
        self.play(
            p_l1.animate.set_color(COLOR_LEFT),
            p_l2.animate.set_color(COLOR_LEFT),
            p_r1.animate.set_color(COLOR_RIGHT),
            p_r2.animate.set_color(COLOR_RIGHT),
            run_time=1
        )
        
        # Add labels for the counts (Resolving Issues 31, 32)
        label_left = Text("Left: 2", font_size=24, color=COLOR_LEFT)
        label_right = Text("Right: 2", font_size=24, color=COLOR_RIGHT)
        self.place_at_grid(label_left, 'A3')
        self.place_at_grid(label_right, 'A5')
        self.play(Write(label_left), Write(label_right))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_PIVOT)
        
        # Setup rotation using ValueTracker for the angle
        angle_tracker = ValueTracker(PI/2) 
        
        # Track the current pivot object to allow for pivot switches
        pivot_ref = [p_pivot]
        
        def update_line(m):
            ang = angle_tracker.get_value()
            center = pivot_ref[0].get_center()
            direction = np.array([np.cos(ang), np.sin(ang), 0])
            # Use put_start_and_end_on for consistent length
            m.put_start_and_end_on(center - 4.5*direction, center + 4.5*direction)
            
        line.add_updater(update_line)
        
        # Target angle to hit p_r1 (at B5)
        vec = p_r1.get_center() - p_pivot.get_center()
        hit_angle = np.arctan2(vec[1], vec[0])
        
        # Rotate clockwise (angle decreasing from PI/2 to hit_angle)
        self.play(angle_tracker.animate.set_value(hit_angle), run_time=2.5, rate_func=linear)
        
        # Hit! Switch pivot
        # "The old green pivot turns blue, and the hit red point turns green."
        old_p = pivot_ref[0]
        new_p = p_r1
        
        self.play(
            old_p.animate.set_color(COLOR_LEFT),
            new_p.animate.set_color(COLOR_PIVOT),
            run_time=0.5
        )
        # Update pivot reference for the updater
        pivot_ref[0] = new_p
        
        # Emphasize the invariant labels
        self.play(Indicate(label_left), Indicate(label_right))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        # Continue rotation slightly to show stability
        self.play(angle_tracker.animate.set_value(hit_angle - 0.4), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        # Final rotation and emphasis on the labels
        self.play(angle_tracker.animate.set_value(hit_angle - 0.8), run_time=2, rate_func=linear)
        self.play(Circumscribe(VGroup(label_left, label_right)))
        self.wait(2)
