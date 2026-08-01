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
        # Title and Lecture Lines
        title = "The Accumulation Function"
        lines = [
            "Define the accumulation function as the area so far.",
            "As the boundary moves, the shaded area grows.",
            "We denote this area function as A(x).",
            "A(x) tracks the total integral from start to x.",
            "Watch how A(x) changes as we scan the curve."
        ]
        self.setup_layout(title, lines)

        # Colors
        YELLOW_C = "#FFFF00"
        WHITE_C = "#FFFFFF"
        BLUE_C = "#0000FF"

        # Math Functions
        def f_func(t):
            return 0.5 * np.sin(t) + 1.2

        def a_func(x):
            # Integral of f_func from 0 to x:
            # \int (0.5*sin(t) + 1.2) dt = -0.5*cos(t) + 1.2*t
            return -0.5 * np.cos(x) + 1.2 * x

        # 1. Setup Axes
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 6, 1],
            axis_config={"include_tip": True, "color": GREY},
            x_length=5.5,
            y_length=5.5
        ).add_coordinates()
        
        self.place_in_area(axes, "A1", "F6", scale_factor=0.8)
        self.add(axes)

        # 2. Objects for Animation
        f_graph = axes.plot(f_func, x_range=[0, 4.5], color=YELLOW_C)
        f_label = MathTex(r"f(t)", color=YELLOW_C)
        # Resolved Issue 27: Move f_label to D6
        self.place_at_grid(f_label, "D6", scale_factor=0.8)

        # Marks
        a_val = 0.5
        x_tracker = ValueTracker(a_val)
        
        a_line = axes.get_vertical_line(axes.c2p(a_val, f_func(a_val)), color=WHITE_C)
        a_tex = MathTex("a", color=WHITE_C).next_to(axes.c2p(a_val, 0), DOWN)
        
        # Area and Scanning Bar
        shaded_area = always_redraw(lambda:
            axes.get_area(
                f_graph,
                x_range=[a_val, x_tracker.get_value()],
                color=YELLOW_C,
                opacity=0.3
            )
        )
        
        scanning_bar = always_redraw(lambda:
            Line(
                axes.c2p(x_tracker.get_value(), 0),
                axes.c2p(x_tracker.get_value(), f_func(x_tracker.get_value())),
                color=WHITE_C,
                stroke_width=4
            )
        )

        # Optimization: Avoid MathTex inside always_redraw (Instruction 11)
        x_tex = MathTex("x", color=WHITE_C)
        x_tex.add_updater(lambda m: m.next_to(axes.c2p(x_tracker.get_value(), 0), DOWN))

        # Accumulation Graph
        # A(x) = \int_a^x f(t) dt
        a_graph = always_redraw(lambda:
            axes.plot(
                lambda t: a_func(t) - a_func(a_val),
                x_range=[a_val, max(a_val + 0.01, x_tracker.get_value())],
                color=BLUE_C
            )
        )
        a_label_tex = MathTex(r"A(x)", color=BLUE_C)
        # Resolved Issue 26: Move a_label_tex to F1
        self.place_at_grid(a_label_tex, "F1", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # "Define the accumulation function as the area so far."
        self.lecture[0].set_color(YELLOW_C)
        self.play(Create(f_graph), Write(f_label))
        self.play(Create(a_line), Write(a_tex))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "As the boundary moves, the shaded area grows."
        self.lecture[1].set_color(WHITE_C)
        self.add(shaded_area, scanning_bar, x_tex)
        self.play(x_tracker.animate.set_value(1.5), run_time=2, rate_func=rate_functions.smooth)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "We denote this area function as A(x)."
        self.lecture[2].set_color(BLUE_C)
        self.play(Write(a_label_tex), Create(a_graph))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "A(x) tracks the total integral from start to x."
        self.lecture[3].set_color(BLUE_C)
        self.play(x_tracker.animate.set_value(3.0), run_time=2, rate_func=rate_functions.smooth)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Watch how A(x) changes as we scan the curve."
        self.lecture[4].set_color(WHITE_C)
        self.play(x_tracker.animate.set_value(4.5), run_time=3, rate_func=rate_functions.smooth)
        self.wait(2)
