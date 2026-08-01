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
        # Initial layout setup
        title = "The Gatekeeper: Activation Functions"
        lines = [
            "Activation functions decide if the signal should pass through.",
            "They introduce non-linearity, allowing for complex pattern recognition.",
            "ReLU squashes negative values to zero and keeps positive ones."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_CURVE = "#00BFFF"
        COLOR_GATE = "#FF4500"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_CURVE))
        
        # Setup Axes on the right side
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 3, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"color": WHITE, "include_tip": True}
        )
        self.place_in_area(axes, "B2", "E6")
        
        linear_graph = axes.plot(lambda x: x, color=COLOR_CURVE, x_range=[-3, 3])
        # Fix Issue #31: Move linear_label to A5, scale 0.8
        linear_label = Text("y = x", color=COLOR_CURVE, font_size=32)
        self.place_at_grid(linear_label, "A5", scale_factor=0.8)

        self.play(Create(axes), Create(linear_graph))
        self.play(Write(linear_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_CURVE))
        
        # Transform Linear to ReLU (hockey stick curve)
        relu_graph = axes.plot(lambda x: max(0, x), color=COLOR_CURVE, x_range=[-3, 3])
        # Fix Issue #32: Move relu_label to A5, scale 0.8
        relu_label = Text("f(z) = max(0, z)", color=COLOR_CURVE, font_size=32)
        self.place_at_grid(relu_label, "A5", scale_factor=0.8)

        self.play(
            Transform(linear_graph, relu_graph),
            Transform(linear_label, relu_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_GATE))

        # Gatekeeper icon
        gatekeeper = VGroup(
            Circle(radius=0.3, color=COLOR_GATE, fill_opacity=0.5),
            Line(start=LEFT*0.2, end=RIGHT*0.2, color=WHITE).rotate(45*DEGREES),
            Line(start=LEFT*0.2, end=RIGHT*0.2, color=WHITE).rotate(-45*DEGREES)
        )
        # Place Gatekeeper near the origin of axes
        gatekeeper.move_to(axes.c2p(0, 0))
        self.play(FadeIn(gatekeeper))

        # Dot and ValueTracker for squashing animation
        z_tracker = ValueTracker(-2.5)
        
        # input dot z
        input_dot = Dot(color=WHITE)
        input_dot.add_updater(lambda d: d.move_to(axes.c2p(z_tracker.get_value(), 0)))
        
        # output dot f(z)
        output_dot = Dot(color=COLOR_GATE)
        output_dot.add_updater(lambda d: d.move_to(axes.c2p(z_tracker.get_value(), max(0, z_tracker.get_value()))))
        
        # labels for dots
        z_label = Text("z", font_size=20, color=WHITE)
        z_label.add_updater(lambda l: l.next_to(input_dot, DOWN, buff=0.1))
        
        out_label = Text("out", font_size=20, color=COLOR_GATE)
        out_label.add_updater(lambda l: l.next_to(output_dot, UP, buff=0.1))

        self.add(input_dot, output_dot, z_label, out_label)

        # Animate from negative to positive
        # Negative part: Gatekeeper blocks
        self.play(z_tracker.animate.set_value(0), run_time=2, rate_func=linear)
        self.wait(0.5)
        
        # Positive part: Signal passes through
        self.play(z_tracker.animate.set_value(2.5), run_time=2, rate_func=linear)
        self.wait(2)
