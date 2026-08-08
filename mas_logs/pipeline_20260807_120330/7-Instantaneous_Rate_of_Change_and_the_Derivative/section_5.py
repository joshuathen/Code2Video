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
        # Initialization
        title = "Formalizing the Derivative"
        lines = [
            "We define this instantaneous rate as the derivative, f prime.",
            "It measures the slope of the tangent at any point.",
            "The formula uses a limit as 'h' approaches zero.",
            "We find the slope over an infinitely small interval.",
            "This powerful tool describes how functions change instantly."
        ]
        self.setup_layout(title, lines)

        # Colors
        GREEN_COLOR = "#00FF00"
        BLUE_COLOR = "#52C1FF"
        YELLOW_COLOR = "#FFFF00"
        WHITE_COLOR = "#FFFFFF"

        # Math Objects - Formal Derivative Formula
        # Breakdown for coloring: f'(x) [0], = [1], lim [2], numerator [3], bar [4], denominator [5]
        formula = MathTex(
            "f'(x)",                      # 0
            "=",                         # 1
            "\\lim_{h \\to 0}",           # 2
            "{f(x+h) - f(x)",             # 3
            "\\over",                    # 4
            "h}",                        # 5
            color=WHITE_COLOR, font_size=36
        )
        # Fix from Issues 38, 44: Move to A2-B5 for prominence and clearance
        self.place_in_area(formula, "A2", "B5", scale_factor=1.0)
        
        # Value Trackers
        h_tracker = ValueTracker(1.5)
        x_tracker = ValueTracker(-0.5)
        
        # Graph Setup
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-1, 4, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "font_size": 20, "color": WHITE_COLOR}
        ).add_coordinates()
        
        graph_area = VGroup(axes)
        # Fix from Issues 39, 44: Re-center to C2-F5 for better balance
        self.place_in_area(graph_area, "C2", "F5", scale_factor=0.8)

        def func(x):
            return 0.5 * x**2 + 0.5

        curve = axes.plot(func, color=WHITE_COLOR)
        
        # Updaters for dynamic elements
        def get_secant_line():
            x0 = x_tracker.get_value()
            h = h_tracker.get_value()
            if abs(h) < 0.001:
                m = x0 # Derivative of 0.5x^2 + 0.5 is x
            else:
                m = (func(x0 + h) - func(x0)) / h
            # Line through (x0, f(x0)) with slope m
            # x_range constraint to stay within visual area bounds
            line = axes.plot(lambda x: m * (x - x0) + func(x0), color=YELLOW_COLOR, x_range=[max(-2, x0 - 1.5), min(2, x0 + h + 1.5)])
            return line

        secant_line = always_redraw(get_secant_line)
        
        point_p = always_redraw(lambda: Dot(axes.c2p(x_tracker.get_value(), func(x_tracker.get_value())), radius=0.06, color=WHITE))
        point_q = always_redraw(lambda: Dot(axes.c2p(x_tracker.get_value() + h_tracker.get_value(), func(x_tracker.get_value() + h_tracker.get_value())), radius=0.06, color=WHITE))

        # Vertical and Horizontal segments for rise/run
        rise_line = always_redraw(lambda: Line(
            axes.c2p(x_tracker.get_value() + h_tracker.get_value(), func(x_tracker.get_value())),
            axes.c2p(x_tracker.get_value() + h_tracker.get_value(), func(x_tracker.get_value() + h_tracker.get_value())),
            color=GREEN_COLOR, stroke_width=4
        ))
        
        run_line = always_redraw(lambda: Line(
            axes.c2p(x_tracker.get_value(), func(x_tracker.get_value())),
            axes.c2p(x_tracker.get_value() + h_tracker.get_value(), func(x_tracker.get_value())),
            color=BLUE_COLOR, stroke_width=4
        ))

        # === Animation for Lecture Line 1 ===
        # We define this instantaneous rate as the derivative, f prime.
        # Animation: The formal derivative formula appears in white (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(WHITE_COLOR))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It measures the slope of the tangent at any point.
        # Animation: Highlight f(x+h) - f(x) in green (#00FF00) on the graph.
        self.play(self.lecture[1].animate.set_color(GREEN_COLOR))
        self.play(Create(axes), Create(curve))
        self.play(Create(point_p), Create(point_q), Create(secant_line))
        # Highlight numerator on formula and rise on graph
        self.play(formula[3].animate.set_color(GREEN_COLOR), Create(rise_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The formula uses a limit as 'h' approaches zero.
        # Animation: The 'h' denominator shrinks as the 'h -> 0' limit is applied.
        self.play(self.lecture[2].animate.set_color(BLUE_COLOR))
        # Highlight denominator on formula and run on graph
        self.play(formula[5].animate.set_color(BLUE_COLOR), Create(run_line))
        self.play(formula[2].animate.set_color(BLUE_COLOR)) # Highlight limit notation
        # Animate h -> 0
        self.play(
            h_tracker.animate.set_value(0.01),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # We find the slope over an infinitely small interval.
        # Animation: The notation 'f prime of x' glows in yellow (#FFFF00).
        self.play(self.lecture[3].animate.set_color(YELLOW_COLOR))
        self.play(
            formula[0].animate.set_color(YELLOW_COLOR),
            Flash(formula[0], color=YELLOW_COLOR, flash_radius=0.5)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This powerful tool describes how functions change instantly.
        # Animation: The tangent line glides smoothly along the entire curve.
        self.play(self.lecture[4].animate.set_color(WHITE_COLOR))
        # Ensure h is tiny for tangent representation
        h_tracker.set_value(0.001)
        self.play(
            x_tracker.animate.set_value(1.0),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
