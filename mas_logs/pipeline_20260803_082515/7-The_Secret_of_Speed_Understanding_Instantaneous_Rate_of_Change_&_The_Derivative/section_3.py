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
        # Fetching content from storyboard
        title_text = "The Curved Path Problem"
        lecture_lines = [
            "Curves represent things that change speed over time.",
            "At any point, the steepness is constantly shifting.",
            "Our standard slope formula needs two points, not one."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Setup visualization elements
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 3, 1],
            axis_config={"include_tip": False, "stroke_width": 2},
            tips=False
        )
        # Function: y = 0.2 * x^2
        curve = axes.plot(lambda x: 0.2 * x**2, color=WHITE)
        
        plot_group = VGroup(axes, curve)
        # Issue 27 Fix applied: Move to B2-E6, scale 0.7 to avoid overlap with notes
        self.place_in_area(plot_group, 'B2', 'E6', scale_factor=0.7)
        
        # Value tracker for car position along x-axis
        t_tracker = ValueTracker(-2.5)

        # === Animation for Lecture Line 1 ===
        # "Curves represent things that change speed over time."
        # A white (#FFFFFF) curve representing a parabola y = 0.2x^2 appears.
        self.lecture[0].set_color(WHITE)
        self.play(Create(axes), Create(curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "At any point, the steepness is constantly shifting."
        # A cyan (#00FFFF) small rectangle moves along the curve to simulate a car.
        self.lecture[1].set_color("#00FFFF")
        
        # Car mobject (Polygon for robustness)
        car = Polygon(
            np.array([-0.2, -0.1, 0]),
            np.array([0.2, -0.1, 0]),
            np.array([0.2, 0.1, 0]),
            np.array([-0.2, 0.1, 0]),
            color="#00FFFF", fill_opacity=0.8, stroke_width=2
        )
        
        def update_car(obj):
            t = t_tracker.get_value()
            center = axes.c2p(t, 0.2 * t**2)
            # Calculate local slope for orientation
            dt = 0.01
            p1 = axes.c2p(t, 0.2 * t**2)
            p2 = axes.c2p(t + dt, 0.2 * (t + dt)**2)
            direction = p2 - p1
            angle = np.arctan2(direction[1], direction[0])
            
            # Point-based rotation to avoid accumulation issues
            w, h = 0.4, 0.2
            base_corners = [
                np.array([-w/2, -h/2, 0]),
                np.array([w/2, -h/2, 0]),
                np.array([w/2, h/2, 0]),
                np.array([-w/2, h/2, 0]),
            ]
            rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            rotated_corners = [center + np.dot(rot_matrix, c) for c in base_corners]
            obj.set_points_as_corners([*rotated_corners, rotated_corners[0]])

        car.add_updater(update_car)
        self.play(FadeIn(car))
        # Animate movement halfway to show progress
        self.play(t_tracker.animate.set_value(0), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Our standard slope formula needs two points, not one."
        # A yellow (#FFFF00) line segment follows the car, showing changing steepness.
        self.lecture[2].set_color("#FFFF00")
        
        tangent_line = Line(color="#FFFF00", stroke_width=5)
        
        def update_tangent(obj):
            t = t_tracker.get_value()
            center = axes.c2p(t, 0.2 * t**2)
            dt = 0.01
            p1 = axes.c2p(t, 0.2 * t**2)
            p2 = axes.c2p(t + dt, 0.2 * (t + dt)**2)
            direction = p2 - p1
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                return
            direction /= direction_norm
            
            # Visually consistent tangent segment length
            line_length = 1.6
            obj.set_points_as_corners([
                center - direction * (line_length / 2),
                center + direction * (line_length / 2)
            ])
            
        tangent_line.add_updater(update_tangent)
        self.play(Create(tangent_line))
        
        # Finish the movement to demonstrate changing slope
        self.play(t_tracker.animate.set_value(2.5), run_time=3, rate_func=linear)
        self.wait(2)
