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
        lecture_lines = [
            "Imagine a moving curtain sweeping across a graph.",
            "Shaded area grows as the curtain moves right.",
            "This growing area defines the accumulation function.",
            "As input x increases, the total area accumulates.",
            "This function maps points to their total area."
        ]
        self.setup_layout("The Accumulation Function", lecture_lines)

        # Colors
        COLOR_F = "#00FF00"  # Green for f(t)
        COLOR_CURTAIN = "#FFFFFF"  # White for vertical line
        COLOR_A = "#FF8C00"  # Orange for A(x)

        # --- Setup Axes and Functions ---
        # Top Axes for f(t)
        top_axes = Axes(
            x_range=[0, 4.5, 1],
            y_range=[0, 2.5, 1],
            x_length=4.0,
            y_length=2.0,
            axis_config={"include_tip": True, "font_size": 18}
        )
        # Issue 32 fix: start from A2 instead of A1 to prevent crowding
        self.place_in_area(top_axes, "A2", "C6", scale_factor=0.8)
        
        # f(t) = 0.3 sin(2t) + 1.2
        def f_func(t):
            return 0.3 * np.sin(2 * t) + 1.2
        
        f_graph = top_axes.plot(f_func, x_range=[0, 4], color=COLOR_F)
        f_label = MathTex("f(t)", color=COLOR_F, font_size=24)
        self.place_at_grid(f_label, "A2", scale_factor=0.8)

        # Bottom Axes for A(x)
        bottom_axes = Axes(
            x_range=[0, 4.5, 1],
            y_range=[0, 6, 2],
            x_length=4.0,
            y_length=2.0,
            axis_config={"include_tip": True, "font_size": 18}
        )
        # Issue 33 fix: start from D2 instead of D1 to maintain margin
        self.place_in_area(bottom_axes, "D2", "F6", scale_factor=0.8)
        
        # A(x) = Integral of f(t) from 0 to x
        # Integral of 0.3 sin(2t) + 1.2 is -0.15 cos(2t) + 1.2t + C
        # At x=0, A(0)=0 => -0.15 cos(0) + 0 + C = 0 => -0.15 + C = 0 => C = 0.15
        def A_func(x):
            return -0.15 * np.cos(2 * x) + 1.2 * x + 0.15

        a_label = MathTex("A(x) = \\int_0^x f(t) dt", color=COLOR_A, font_size=24)
        # Issue 34 fix: scale factor 0.7 at D1 to avoid crowding lecture notes
        self.place_at_grid(a_label, "D1", scale_factor=0.7)

        # State Tracker
        x_tracker = ValueTracker(0)

        # Curtain Line
        curtain = Line(
            top_axes.c2p(0, 0), top_axes.c2p(0, 2.2),
            color=COLOR_CURTAIN, stroke_width=2
        )
        curtain.add_updater(lambda m: m.move_to(top_axes.c2p(x_tracker.get_value(), 1.1)))

        # Area Mobject (using Polygon for accumulation)
        area = VMobject(fill_color=COLOR_F, fill_opacity=0.4, stroke_width=0)
        def update_area(m):
            x_val = x_tracker.get_value()
            if x_val <= 0.01:
                m.set_points_as_corners([top_axes.c2p(0,0), top_axes.c2p(0.01, 0)])
            else:
                points = [top_axes.c2p(0, 0)]
                t_vals = np.linspace(0, x_val, 30)
                points.extend([top_axes.c2p(t, f_func(t)) for t in t_vals])
                points.append(top_axes.c2p(x_val, 0))
                m.set_points_as_corners(points)
        area.add_updater(update_area)

        # Orange trace for A(x)
        a_trace = VMobject(color=COLOR_A, stroke_width=3)
        def update_trace(m):
            x_val = x_tracker.get_value()
            if x_val <= 0.01:
                m.set_points_as_corners([bottom_axes.c2p(0,0), bottom_axes.c2p(0.01,0)])
            else:
                t_vals = np.linspace(0, x_val, 30)
                m.set_points_as_corners([bottom_axes.c2p(t, A_func(t)) for t in t_vals])
        a_trace.add_updater(update_trace)

        # === Animation for Lecture Line 1 ===
        # Imagine a moving curtain sweeping across a graph.
        self.lecture[0].set_color(YELLOW)
        self.play(
            Create(top_axes),
            Create(f_graph),
            Write(f_label),
            Create(curtain),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Shaded area grows as the curtain moves right.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.add(area)
        self.play(
            x_tracker.animate.set_value(1.5),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This growing area defines the accumulation function.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(
            Create(bottom_axes),
            Write(a_label),
            run_time=2
        )
        self.add(a_trace)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # As input x increases, the total area accumulates.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(
            x_tracker.animate.set_value(3.5),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This function maps points to their total area.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Dot on A(x) and connecting line logic
        dot = Dot(bottom_axes.c2p(3.5, A_func(3.5)), color=COLOR_A)
        v_line = DashedLine(
            top_axes.c2p(3.5, 0),
            bottom_axes.c2p(3.5, A_func(3.5)),
            color=WHITE,
            stroke_width=1
        )
        
        self.play(
            FadeIn(dot),
            Create(v_line),
            run_time=1.5
        )
        self.wait(2)

        # Reset colors
        self.lecture[4].set_color(WHITE)
        self.wait(2)
