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
        # Define lecture lines
        lecture_lines = [
            "- Meet Willa and her rotating laser beam line.",
            "- She starts at an initial point on the plane.",
            "- The beam rotates clockwise around this pivot point.",
            "- When hitting another point, the pivot immediately switches.",
            "- This creates the \"Windmill\" motion through the points."
        ]
        
        # Setup layout
        self.setup_layout("Introduction to 'Willa the Windmill'", lecture_lines)
        
        # Points positions (Resolving Issues 25 & 26: Positioning P at C2 and Q at D5)
        dot_p = Dot(color=WHITE)
        self.place_at_grid(dot_p, 'C2', scale_factor=0.8)
        pos_p = dot_p.get_center().copy()
        
        dot_q = Dot(color=WHITE)
        self.place_at_grid(dot_q, 'D5', scale_factor=0.8)
        pos_q = dot_q.get_center().copy()
        
        dots_others = VGroup()
        for pos_key in ['A4', 'B6', 'F1']:
            d = Dot(color=WHITE)
            self.place_at_grid(d, pos_key, scale_factor=0.8)
            dots_others.add(d)
            
        all_dots = VGroup(dot_p, dot_q, dots_others)
        
        # Labels - positioned close to dots (within 1 grid unit)
        label_p = Text("P", font_size=20, color=WHITE).next_to(dot_p, UP, buff=0.1)
        label_q = Text("Q", font_size=20, color=WHITE).next_to(dot_q, DOWN, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        # Fade in 5 white points (#FFFFFF). Label one 'P'.
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(all_dots), FadeIn(label_p))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # She starts at an initial point. Draw a yellow line (#FFFF00) passing through point 'P'.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Pivot tracking mobject
        pivot_tracker = Dot(pos_p).set_opacity(0)
        self.add(pivot_tracker)
        
        # Angle tracker (initial angle: 60 degrees)
        angle_tracker = ValueTracker(60 * DEGREES)
        
        # Laser beam line - Resolving Issue 24 (clipping line to grid area to avoid obstructing lecture)
        grid_bounds = [0.4, 5.6, -2.9, 2.3] # x_min, x_max, y_min, y_max

        def get_line_endpoints(p, angle, bounds):
            x_min, x_max, y_min, y_max = bounds
            ux, uy = np.cos(angle), np.sin(angle)
            t_candidates = []
            if abs(ux) > 1e-6:
                t_candidates.append((x_min - p[0]) / ux)
                t_candidates.append((x_max - p[0]) / ux)
            if abs(uy) > 1e-6:
                t_candidates.append((y_min - p[1]) / uy)
                t_candidates.append((y_max - p[1]) / uy)
            
            t_pos = [t for t in t_candidates if t > 0]
            t_neg = [t for t in t_candidates if t < 0]
            # Use intersection points with the grid boundaries
            t_start = min(t_pos) if t_pos else 6
            t_end = max(t_neg) if t_neg else -6
            return p + t_start * np.array([ux, uy, 0]), p + t_end * np.array([ux, uy, 0])

        laser_line = Line(color="#FFFF00")
        laser_line.add_updater(lambda mob: mob.put_start_and_end_on(
            *get_line_endpoints(pivot_tracker.get_center(), angle_tracker.get_value(), grid_bounds)
        ))
        
        self.play(Create(laser_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The beam rotates clockwise around this pivot point.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Calculate target angle to hit Q
        vec_pq = pos_q - pos_p
        target_angle = np.arctan2(vec_pq[1], vec_pq[0])
        
        # Ensure clockwise rotation (angle decreases)
        current_angle = angle_tracker.get_value()
        while target_angle > current_angle:
            target_angle -= 2 * PI
            
        self.play(angle_tracker.animate.set_value(target_angle), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # When hitting another point, the pivot immediately switches.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Highlight hitting point Q
        self.play(
            dot_q.animate.set_color("#FFFF00").scale(1.5),
            FadeIn(label_q)
        )
        self.play(dot_q.animate.scale(1/1.5))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Change 'P' to white and set the green color (#2ECC71) to 'Q' as the new pivot.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(
            dot_p.animate.set_color(WHITE),
            dot_q.animate.set_color("#2ECC71"),
            pivot_tracker.animate.move_to(pos_q)
        )
        
        # Continue rotating clockwise (further decrease angle)
        self.play(angle_tracker.animate.increment_value(-60 * DEGREES), run_time=2, rate_func=linear)
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
