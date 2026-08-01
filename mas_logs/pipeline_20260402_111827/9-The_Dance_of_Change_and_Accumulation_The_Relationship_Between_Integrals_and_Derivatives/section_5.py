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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        lecture_lines = [
            "Let's look at the original function curve.",
            "We shade the area as we move right.",
            "The top graph tracks total accumulated area.",
            "Area growth rate matches the curve's height.",
            "The derivative of area is the function."
        ]
        self.setup_layout("Visualizing the Link: The Area Function", lecture_lines)

        # Define functions
        def f_func(t):
            return 0.3 * t + 0.4
        
        def a_func(x):
            return 0.15 * (x**2) + 0.4 * x

        # Colors
        COLOR_F = "#32CD32"  # Lime green
        COLOR_AREA = "#87CEEB" # Light blue
        COLOR_A = "#FF4500"  # Orange-red
        COLOR_FORMULA = "#FFFFFF"

        # 1. Setup Axes
        # Issue 39 Fix: self.place_in_area(axes_bottom, 'E1', 'F5', scale_factor=0.7)
        axes_bottom = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 2, 1],
            x_length=5,
            y_length=2,
            axis_config={"include_tip": False, "font_size": 16}
        ).set_color(GRAY)
        self.place_in_area(axes_bottom, 'E1', 'F5', scale_factor=0.7)

        # Issue 38 Fix: self.place_in_area(axes_top, 'A2', 'C5', scale_factor=0.7)
        axes_top = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=2,
            axis_config={"include_tip": False, "font_size": 16}
        ).set_color(GRAY)
        self.place_in_area(axes_top, 'A2', 'C5', scale_factor=0.7)

        # 2. Labels (Issue 40 Fix: restriction to column 5 to pull labels inward)
        label_f = Text("f(t)", color=COLOR_F, font_size=20)
        self.place_at_grid(label_f, 'E5', scale_factor=1.0)
        
        label_A = Text("A(x)", color=COLOR_A, font_size=20)
        self.place_at_grid(label_A, 'A5', scale_factor=1.0)

        # Formula - Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        formula = Text("A'(x) = f(x)", color=COLOR_FORMULA, font_size=32)
        self.place_at_grid(formula, "D5", scale_factor=1.0)

        # 3. Dynamic Elements
        f_graph = axes_bottom.plot(f_func, x_range=[0, 4], color=COLOR_F)
        x_tracker = ValueTracker(0.0001)

        # Area under f(t)
        area = always_redraw(lambda: axes_bottom.get_area(
            f_graph, 
            x_range=[0, x_tracker.get_value()], 
            color=COLOR_AREA, 
            opacity=0.5
        ))

        # Vertical scanner line
        scanner = always_redraw(lambda: Line(
            axes_bottom.c2p(x_tracker.get_value(), 0),
            axes_bottom.c2p(x_tracker.get_value(), f_func(x_tracker.get_value())),
            color=COLOR_AREA,
            stroke_width=2
        ))

        # Trace for A(x) graph
        A_trace = always_redraw(lambda: axes_top.plot(
            a_func, 
            x_range=[0.0001, max(0.0001, x_tracker.get_value())], 
            color=COLOR_A
        ))

        # Dot tracking on A(x)
        dot_A = always_redraw(lambda: Dot(
            axes_top.c2p(x_tracker.get_value(), a_func(x_tracker.get_value())),
            color=COLOR_A,
            radius=0.06
        ))

        # Visual link between height of f(t) and height of A(x)
        link_line = always_redraw(lambda: DashedLine(
            axes_bottom.c2p(x_tracker.get_value(), f_func(x_tracker.get_value())),
            axes_top.c2p(x_tracker.get_value(), a_func(x_tracker.get_value())),
            color=WHITE,
            stroke_opacity=0.4
        ))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_F), run_time=0.1)
        self.play(Create(axes_bottom), Create(f_graph), Write(label_f))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_AREA), run_time=0.1)
        self.add(area, scanner)
        self.play(x_tracker.animate.set_value(1.0), run_time=1.5, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_A), run_time=0.1)
        self.play(Create(axes_top), Write(label_A))
        self.add(A_trace, dot_A)
        self.play(x_tracker.animate.set_value(2.5), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE), run_time=0.1)
        self.add(link_line)
        self.play(x_tracker.animate.set_value(4.0), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_FORMULA), run_time=0.1)
        self.play(Write(formula))
        self.wait(2)
