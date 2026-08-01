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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with updated lecture lines
        self.setup_layout(
            "Practical Application: The Cheetah's Sprint", 
            [
                "We start with the cheetah’s recorded acceleration data.",
                "Integrating acceleration gives the cheetah’s changing velocity.",
                "A second integration calculates the total distance covered."
            ]
        )
        
        # Color definitions
        ACCEL_COLOR = "#F91717"
        VELOC_COLOR = "#58C4DD"
        DIST_COLOR = WHITE

        # Function definitions for the sprint (physics-based)
        # a(t) = 3 * e^(-0.8t)
        # v(t) = 3.75 * (1 - e^(-0.8t))
        # s(t) = 3.75 * (t + 1.25 * e^(-0.8t) - 1.25)
        def a_func(t): return 3 * np.exp(-0.8 * t)
        def v_func(t): return (3 / 0.8) * (1 - np.exp(-0.8 * t))
        def s_func(t): return (3 / 0.8) * (t + (1 / 0.8) * np.exp(-0.8 * t) - (1 / 0.8))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ACCEL_COLOR)
        
        # Create axes for a(t) and v(t) - Reduced area to avoid clutter (Issue 51)
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, "B1", "D5", scale_factor=0.8)
        
        labels = axes.get_axis_labels(
            x_label=Text("t", font_size=20), 
            y_label=Text("y", font_size=20)
        )
        a_graph = axes.plot(a_func, x_range=[0, 4], color=ACCEL_COLOR)
        a_label = Text("a(t)", color=ACCEL_COLOR, font_size=24)
        # Moved label to column 6 to avoid overlap (Issue 49)
        self.place_at_grid(a_label, "B6", scale_factor=0.8)

        self.play(Create(axes), Create(labels))
        self.play(Create(a_graph), Write(a_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(VELOC_COLOR)
        
        # Area under acceleration graph represents change in velocity
        area = axes.get_area(a_graph, x_range=[0, 4], color=VELOC_COLOR, opacity=0.3)
        v_graph = axes.plot(v_func, x_range=[0, 4], color=VELOC_COLOR)
        v_label = Text("v(t)", color=VELOC_COLOR, font_size=24)
        # Moved label to column 6 to avoid obscuring graph (Issue 50)
        self.place_at_grid(v_label, "C6", scale_factor=0.8)

        self.play(FadeIn(area))
        self.play(Create(v_graph), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(DIST_COLOR)
        
        # Create a track for the cheetah in the bottom area
        track_line = Line(start=LEFT*2.5, end=RIGHT*2.5, color=GRAY_D)
        track_group = VGroup(track_line)
        self.place_in_area(track_group, "E1", "F6", scale_factor=0.8)
        
        cheetah_dot = Dot(color=YELLOW)
        cheetah_dot.move_to(track_line.get_start())
        
        dist_text = Text("Distance:", font_size=20).next_to(track_line, UP, buff=0.2)
        # Using mob_class=Text to maintain compatibility
        dist_value = DecimalNumber(0, num_decimal_places=2, font_size=20, mob_class=Text).next_to(dist_text, RIGHT)
        dist_group = VGroup(dist_text, dist_value)
        
        self.play(Create(track_line), FadeIn(cheetah_dot), Write(dist_group))
        
        # Use ValueTracker to animate movement and distance accumulation
        time_tracker = ValueTracker(0)
        
        # Updaters for synchronization
        cheetah_dot.add_updater(
            lambda m: m.move_to(
                track_line.point_from_proportion(s_func(time_tracker.get_value()) / s_func(4))
            )
        )
        dist_value.add_updater(
            lambda m: m.set_value(s_func(time_tracker.get_value()))
        )
        
        self.play(time_tracker.animate.set_value(4), run_time=4, rate_func=linear)
        
        # Cleanup
        cheetah_dot.clear_updaters()
        dist_value.clear_updaters()
        self.wait(2)
