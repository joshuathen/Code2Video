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
        # Initial setup with the correct lecture lines
        self.setup_layout("Prerequisite: The Linear Equation as a Slider", [
            "- Every neuron uses the basic formula: y = wx + b",
            "- The weight, w, determines signal strength/sensitivity",
            "- The bias, b, sets the threshold for a decision"
        ])

        # Colors for consistency
        color_w = BLUE_B
        color_b = GREEN_B
        color_line = YELLOW

        # Trackers for the variables
        w_tracker = ValueTracker(1.0)
        b_tracker = ValueTracker(0.0)

        # === Animation for Lecture Line 1 ===
        # Formula y = wx + b
        formula = MathTex("y", "=", "w", "x", "+", "b", font_size=42)
        formula.set_color_by_tex("w", color_w)
        formula.set_color_by_tex("b", color_b)
        self.place_in_area(formula, "A3", "A5")
        
        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            Write(formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Weight Slider and Axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=3.5,
            axis_config={"include_tip": False, "color": GRAY_C}
        )
        self.place_in_area(axes, "C3", "E5")
        
        # The Line y = wx + b
        # Rule 10: Use persistent mobjects + updaters
        # Fixed: Changed get_graph to plot for Manim CE v0.19.0 compatibility
        line_graph = axes.plot(lambda x: w_tracker.get_value() * x + b_tracker.get_value(), color=color_line)
        line_graph.add_updater(
            lambda m: m.become(
                axes.plot(lambda x: w_tracker.get_value() * x + b_tracker.get_value(), color=color_line)
            )
        )

        # Weight Slider Visual
        w_line = NumberLine(x_range=[-2, 2, 1], length=2, color=color_w, include_tip=False)
        w_slider_label = MathTex("w", color=color_w, font_size=32)
        w_slider_group = VGroup(w_line, w_slider_label).arrange(LEFT, buff=0.2)
        self.place_in_area(w_slider_group, "B2", "B3", scale_factor=0.8)
        
        w_dot = Dot(color=color_w)
        w_dot.add_updater(lambda d: d.move_to(w_line.n2p(w_tracker.get_value())))

        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_w),
            Create(axes),
            FadeIn(w_slider_group),
            FadeIn(w_dot)
        )
        self.play(Create(line_graph))
        
        # Animate weight change
        self.play(w_tracker.animate.set_value(2.0), run_time=1.5)
        self.play(w_tracker.animate.set_value(-1.0), run_time=2)
        self.play(w_tracker.animate.set_value(1.0), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Bias Slider
        b_line = NumberLine(x_range=[-2, 2, 1], length=2, color=color_b, include_tip=False)
        b_slider_label = MathTex("b", color=color_b, font_size=32)
        b_slider_group = VGroup(b_line, b_slider_label).arrange(LEFT, buff=0.2)
        self.place_in_area(b_slider_group, "B4", "B5", scale_factor=0.8)
        
        b_dot = Dot(color=color_b)
        b_dot.add_updater(lambda d: d.move_to(b_line.n2p(b_tracker.get_value())))

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_b),
            FadeIn(b_slider_group),
            FadeIn(b_dot)
        )
        
        # Animate bias change
        self.play(b_tracker.animate.set_value(1.5), run_time=1.5)
        self.play(b_tracker.animate.set_value(-1.5), run_time=2)
        self.play(b_tracker.animate.set_value(0.0), run_time=1.5)
        
        self.wait(3)
        self.play(self.lecture[2].animate.set_color(WHITE))
