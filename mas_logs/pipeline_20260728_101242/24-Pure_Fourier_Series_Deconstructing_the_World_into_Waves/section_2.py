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

class Section2Scene(TeachingScene):
    def construct(self):
        title_text = "Prerequisite: The Building Blocks (Unit Circles)"
        lecture_lines = [
            "Circular motion naturally creates smooth sine and cosine waves.",
            "A point rotating at constant speed traces a wave.",
            "The fundamental frequency sets the base speed of motion."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_CIRCLE = WHITE
        COLOR_WAVE = "#FF0000"
        COLOR_POINT = YELLOW

        # === Animation for Lecture Line 1 ===
        # Draw a circle #FFFFFF and a point rotating on its edge.
        self.lecture[0].set_color(COLOR_CIRCLE)
        
        # Visual anchor: Circle at C2
        circle = Circle(radius=0.7, color=COLOR_CIRCLE)
        center_point = Dot(radius=0.05, color=WHITE)
        circle_group = VGroup(circle, center_point)
        self.place_at_grid(circle_group, "C2")
        
        radius_line = Line(circle.get_center(), circle.get_right(), color=GRAY)
        rotating_dot = Dot(circle.get_right(), radius=0.08, color=COLOR_POINT)
        
        # Rotation mechanics
        theta = ValueTracker(0)
        
        def update_dot(dot):
            angle = theta.get_value()
            dot.move_to(circle.get_center() + circle.radius * np.array([np.cos(angle), np.sin(angle), 0]))

        def update_radius(line):
            line.set_points_as_corners([circle.get_center(), rotating_dot.get_center()])

        rotating_dot.add_updater(update_dot)
        radius_line.add_updater(update_radius)

        self.play(Create(circle), Create(center_point), FadeIn(radius_line), FadeIn(rotating_dot))
        self.play(theta.animate.set_value(TAU), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Trace a red (#FF0000) sine curve as the point rotates.
        self.lecture[1].set_color(COLOR_WAVE)

        # Start of wave area: next to circle (C3)
        wave_start_x = self.grid["C3"][0] - 0.5
        
        # Line connecting dot to wave
        connection_line = Line(rotating_dot.get_center(), [wave_start_x, rotating_dot.get_y(), 0], color=GRAY, stroke_width=1)
        
        def update_connection(line):
            line.set_points_as_corners([rotating_dot.get_center(), [wave_start_x, rotating_dot.get_y(), 0]])
        
        connection_line.add_updater(update_connection)
        
        # Sine wave path
        wave_points = []
        wave_path = VMobject(color=COLOR_WAVE)
        # Initialize with start point
        start_y = circle.get_center()[1]
        wave_path.set_points_as_corners([[wave_start_x, start_y, 0], [wave_start_x, start_y, 0]])
        
        # Reset theta and prepare for tracing
        theta.set_value(0)
        
        def update_wave(path):
            angle = theta.get_value()
            if angle == 0: return
            # New point based on current Y of rotating dot
            current_y = circle.get_center()[1] + circle.radius * np.sin(angle)
            # Map theta (0 to 2*TAU) to x distance
            current_x = wave_start_x + (angle / (2 * TAU)) * 3.5
            
            # Optimization: avoid too many points
            if len(wave_points) == 0 or np.linalg.norm(np.array([current_x, current_y, 0]) - wave_points[-1]) > 0.05:
                wave_points.append(np.array([current_x, current_y, 0]))
                path.set_points_as_corners(wave_points)

        wave_path.add_updater(update_wave)
        
        self.add(connection_line, wave_path)
        self.play(theta.animate.set_value(2 * TAU), run_time=4, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Label the horizontal distance for one cycle as 'Period (T)'.
        self.lecture[2].set_color(YELLOW)

        # Stop updaters before creating static final labels
        rotating_dot.clear_updaters()
        radius_line.clear_updaters()
        connection_line.clear_updaters()
        wave_path.clear_updaters()

        # One cycle is 1.75 units in the mapping (3.5 / 2)
        p1 = [wave_start_x, circle.get_center()[1] - 0.8, 0]
        p2 = [wave_start_x + 1.75, circle.get_center()[1] - 0.8, 0]
        
        period_brace = BraceBetweenPoints(p1, p2, color=YELLOW)
        period_label = MathTex("T", color=YELLOW).next_to(period_brace, DOWN, buff=0.1)
        period_text = Text("Period (T)", font_size=18, color=YELLOW).next_to(period_label, RIGHT, buff=0.2)
        
        # Group and position label at E4 to fix Issue 24 & 25
        period_label_group = VGroup(period_brace, period_label, period_text)
        self.place_at_grid(period_label_group, 'E4', scale_factor=0.9)

        self.play(FadeIn(period_label_group))
        self.wait(2)
