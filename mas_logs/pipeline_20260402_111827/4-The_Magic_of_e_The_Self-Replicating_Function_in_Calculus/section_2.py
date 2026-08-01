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
        # Initialize Scene
        lecture_lines = [
            "Derivatives measure the steepness of a curve.",
            "For y equals x squared, the slope is 2x.",
            "This rate of change is the tangent's slope."
        ]
        self.setup_layout("Prerequisite Knowledge: Slopes and Derivatives", lecture_lines)

        # ValueTracker for movement
        x_tracker = ValueTracker(-1.2)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=1)

        # Create Axes and Curve (Fix Issue 40 & 42: Positioning and scaling)
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 4, 1],
            axis_config={"color": WHITE, "include_tip": True},
            tips=True
        )
        curve = axes.plot(lambda x: x**2, x_range=[-2, 2], color="#0000FF")
        plot_group = VGroup(axes, curve)
        self.place_in_area(plot_group, "A2", "F5", scale_factor=0.6)

        # Fix Issue 41: Label at C6
        curve_label = Text("y = x^2", color="#0000FF", font_size=24)
        self.place_at_grid(curve_label, "C6", scale_factor=0.7)

        # Dot moving along the curve
        dot = Dot(color="#FFFFFF")
        dot.add_updater(lambda d: d.move_to(axes.c2p(x_tracker.get_value(), x_tracker.get_value()**2)))

        self.play(Create(axes), Create(curve), FadeIn(curve_label), FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00"),
            run_time=1
        )

        # Yellow Tangent Line
        def get_tangent_line():
            x = x_tracker.get_value()
            slope = 2 * x
            p1 = axes.c2p(x - 0.6, (x - 0.6 - x) * slope + x**2)
            p2 = axes.c2p(x + 0.6, (x + 0.6 - x) * slope + x**2)
            return Line(p1, p2, color="#FFFF00")

        tangent_line = always_redraw(get_tangent_line)
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=1
        )

        # Dynamic Slope Label - Use mob_class=Text to avoid LaTeX dependency (FileNotFoundError: latex)
        slope_val = DecimalNumber(num_decimal_places=2, color="#FFFF00", mob_class=Text).scale(0.8)
        slope_prefix = Text("m = ", color="#FFFF00", font_size=20)
        slope_label = VGroup(slope_prefix, slope_val).arrange(RIGHT, buff=0.1)
        
        # Updaters for the label
        slope_val.add_updater(lambda v: v.set_value(2 * x_tracker.get_value()))
        slope_label.add_updater(lambda m: m.next_to(dot, UR, buff=0.1))

        self.play(FadeIn(slope_label))
        self.wait(0.5)

        # Movement animation
        self.play(x_tracker.animate.set_value(1.2), run_time=4, rate_func=linear)
        self.wait(2)
