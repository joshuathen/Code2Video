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
        # Setup the title and lecture lines
        title_text = "Prerequisite: The Circle-Wave Connection"
        lecture_lines = [
            "- Every sine wave correlates to circular motion.",
            "- Imagine a point spinning around a circle at constant speed.",
            "- Tracking the vertical height over time creates a wave."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_CIRCLE = WHITE
        COLOR_WAVE = "#E74C3C"  # Red

        # === Animation for Lecture Line 1 ===
        # Every sine wave correlates to circular motion.
        # Draw a white circle (#FFFFFF) on the left with a rotating dot.
        self.lecture[0].set_color(COLOR_CIRCLE)
        
        circle = Circle(radius=1.2, color=COLOR_CIRCLE)
        # Fix: Move circle to C1 to allow more room for the wave (Issue #50)
        self.place_at_grid(circle, "C1", scale_factor=0.8)
        
        dot_tracker = ValueTracker(0)
        dot = Dot(color=COLOR_CIRCLE)
        # Using add_updater for dot rotation to follow circle's circumference
        dot.add_updater(lambda d: d.move_to(
            circle.get_center() + circle.radius * circle.get_p_meters()[0] * np.array([np.cos(dot_tracker.get_value()), np.sin(dot_tracker.get_value()), 0])
        ))
        # Note: circle.get_p_meters()[0] is just a hack to get scale if needed, 
        # but circle.radius is already scaled because place_at_grid scales the object.
        # Let's simplify the updater logic:
        dot.remove_updater(dot.updaters[0])
        dot.add_updater(lambda d: d.move_to(
            circle.point_at_angle(dot_tracker.get_value())
        ))
        
        self.play(Create(circle), FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imagine a point spinning around a circle at constant speed.
        # A red line (#E74C3C) extends from the dot's height to the right.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_WAVE)

        # Horizontal line connecting dot to the wave starting point
        # Now starting the wave at grid C2 since circle moved to C1
        wave_start_x = self.grid["C2"][0]
        
        connection_line = Line(
            start=dot.get_center(),
            end=[wave_start_x, dot.get_center()[1], 0],
            color=COLOR_WAVE,
            stroke_width=2
        )
        
        connection_line.add_updater(lambda l: l.put_start_and_end_on(
            dot.get_center(),
            [wave_start_x, dot.get_center()[1], 0]
        ))
        
        self.play(Create(connection_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Tracking the vertical height over time creates a wave.
        # The moving line traces a smooth red sine wave as the dot rotates.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_WAVE)

        # We trace the point at the end of the connection line as it moves right
        time_tracker = ValueTracker(0)
        tracing_dot = Dot(color=COLOR_WAVE, radius=0.01) # Small point to trace
        
        # The tracing dot moves horizontally to the right while its Y matches the rotating dot's Y
        tracing_dot.add_updater(lambda d: d.move_to(
            [wave_start_x + time_tracker.get_value() * 1.5, dot.get_center()[1], 0]
        ))
        
        # TracedPath for drawing the sine wave as tracing_dot moves
        wave_path = TracedPath(tracing_dot.get_center, stroke_color=COLOR_WAVE, stroke_width=4)
        
        self.add(tracing_dot, wave_path)
        
        # Animate the rotation and the wave tracing
        # Rotate for two full cycles (4*PI)
        self.play(
            dot_tracker.animate.set_value(4 * PI),
            time_tracker.animate.set_value(2.5), # Slightly more runway for visualization
            run_time=6,
            rate_func=linear
        )
        
        self.wait(2)
