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
        # Setup
        title_text = "Visualizing the Accumulation Function"
        lecture_lines = [
            "Imagine area building up as we move along time.",
            "This growing area creates a new 'accumulation function'.",
            "The rate of this growth is the original curve."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        ORANGE = "#FFA500"
        BLUE = "#ADD8E6"
        GREEN = "#00FF00"
        
        # Trackers
        time_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        # "Imagine area building up as we move along time."
        self.lecture[0].set_color(BLUE)
        
        # Velocity Axes (Bottom Area: D4 to F6)
        # Shifted to columns 4-6 to create gutter (Issue 37)
        vel_axes = Axes(
            x_range=[0, 4, 1], 
            y_range=[0, 2, 1], 
            x_length=3.0, 
            y_length=1.8,
            axis_config={"include_tip": True, "font_size": 16}
        )
        vel_label = Text("Velocity (v)", font_size=18, color=ORANGE)
        vel_group = VGroup(vel_axes, vel_label).arrange(UP, buff=0.1)
        self.place_in_area(vel_group, "D4", "F6")
        
        # Velocity function: v(t) = 1.5
        v_func = vel_axes.plot(lambda t: 1.5, x_range=[0, 3.5], color=ORANGE)
        
        # Area under velocity (persistent)
        area_poly = VMobject(fill_opacity=0.5, color=BLUE, stroke_width=0)
        def update_area(p):
            t = time_tracker.get_value()
            safe_t = max(t, 0.001)
            p.set_points_as_corners([
                vel_axes.c2p(0, 0),
                vel_axes.c2p(0, 1.5),
                vel_axes.c2p(safe_t, 1.5),
                vel_axes.c2p(safe_t, 0)
            ])
        area_poly.add_updater(update_area)

        # Slider asset (Issue 20)
        slider = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg")
        slider.scale(0.15).set_color(WHITE)
        slider.add_updater(lambda s: s.move_to(vel_axes.c2p(time_tracker.get_value(), 0)))

        self.play(Create(vel_axes), Write(vel_label))
        self.play(Create(v_func))
        self.add(area_poly, slider)
        self.play(time_tracker.animate.set_value(2.5), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "This growing area creates a new 'accumulation function'."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        # Distance Axes (Top Area: A4 to C6)
        # Shifted to columns 4-6 to create gutter (Issue 37)
        dist_axes = Axes(
            x_range=[0, 4, 1], 
            y_range=[0, 6, 2], 
            x_length=3.0, 
            y_length=1.8,
            axis_config={"include_tip": True, "font_size": 16}
        )
        dist_label = Text("Distance (s)", font_size=18, color=GREEN)
        dist_group = VGroup(dist_axes, dist_label).arrange(UP, buff=0.1)
        self.place_in_area(dist_group, "A4", "C6")
        
        # Distance graph (s(t) = 1.5 * t)
        dist_line = VMobject(color=GREEN, stroke_width=3)
        def update_dist_line(l):
            t = time_tracker.get_value()
            if t < 0.01:
                l.set_points_as_corners([dist_axes.c2p(0, 0), dist_axes.c2p(0.01, 1.5*0.01)])
            else:
                l.set_points_as_corners([dist_axes.c2p(x, 1.5*x) for x in np.linspace(0, t, 30)])
        dist_line.add_updater(update_dist_line)

        # Vehicle asset (Issue 20)
        vehicle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vehicle.svg")
        vehicle.scale(0.2).set_color(GREEN)
        vehicle.add_updater(lambda v: v.move_to(dist_axes.c2p(time_tracker.get_value(), 1.5 * time_tracker.get_value()) + UP*0.25))

        # Numerical indicators moved to column 3 (Issue 38)
        area_val_label = VGroup(
            Text("Area:", font_size=16, color=BLUE),
            DecimalNumber(0, font_size=16, color=BLUE)
        ).arrange(RIGHT, buff=0.1)
        
        dist_val_label = VGroup(
            Text("Dist:", font_size=16, color=GREEN),
            DecimalNumber(0, font_size=16, color=GREEN)
        ).arrange(RIGHT, buff=0.1)
        
        self.place_at_grid(area_val_label, "E3", scale_factor=0.8)
        self.place_at_grid(dist_val_label, "B3", scale_factor=0.8)
        
        # Update values efficiently
        area_val_label[1].add_updater(lambda d: d.set_value(1.5 * time_tracker.get_value()))
        dist_val_label[1].add_updater(lambda d: d.set_value(1.5 * time_tracker.get_value()))
        
        self.play(Create(dist_axes), Write(dist_label))
        self.add(dist_line, vehicle, area_val_label, dist_val_label)
        
        # Reset and synchronized growth
        self.play(time_tracker.animate.set_value(0), run_time=1)
        self.play(time_tracker.animate.set_value(3.5), run_time=5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The rate of this growth is the original curve."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORANGE)
        
        # Connecting vertical line to highlight relationship
        conn_line = Line(ORIGIN, UP, color=WHITE, stroke_width=1).set_stroke(opacity=0.4)
        def update_conn_line(l):
            t = time_tracker.get_value()
            p1 = dist_axes.c2p(t, 1.5 * t)
            p2 = vel_axes.c2p(t, 1.5)
            l.set_points_as_corners([p1, p2])
        conn_line.add_updater(update_conn_line)
        
        self.add(conn_line)
        
        # Final scrub to demonstrate the rate of growth connection
        self.play(time_tracker.animate.set_value(1.0), run_time=2)
        self.play(time_tracker.animate.set_value(3.0), run_time=2)
        
        self.wait(2)
