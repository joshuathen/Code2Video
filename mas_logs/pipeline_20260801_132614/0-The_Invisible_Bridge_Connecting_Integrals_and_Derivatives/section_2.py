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
        # Fetching storyboard data
        title = "Prerequisite Review: The Derivative as a 'Rate'"
        lines = [
            "Derivatives measure the instantaneous rate of change.",
            "We visualize this as the slope of a curve.",
            "It tells us how fast a value is changing."
        ]
        self.setup_layout(title, lines)
        
        # Define colors for synchronization
        CURVE_COLOR = "#FFFFFF"
        TANGENT_COLOR = "#FFFF00"
        SLOPE_COLOR = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        # Line 1: Derivatives measure the instantaneous rate of change.
        self.play(self.lecture[0].animate.set_color(CURVE_COLOR))
        
        # Area B2 to E5 for axes and curve
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-1, 4, 1],
            axis_config={"include_tip": False, "color": GREY},
            x_length=3.5,
            y_length=3.5
        )
        self.place_in_area(axes, 'B2', 'E5')
        
        # Curve: f(x) = 0.5x^2 + 1
        curve = axes.plot(lambda x: 0.5 * x**2 + 1, color=CURVE_COLOR)
        
        self.play(Create(axes), Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: We visualize this as the slope of a curve.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(TANGENT_COLOR)
        )
        
        # Tracker for x-coordinate to drive the point and tangent line
        x_tracker = ValueTracker(-1.2)
        
        # Tangent line updated in place
        tangent = Line(color=TANGENT_COLOR)
        def update_tangent(line):
            x0 = x_tracker.get_value()
            y0 = 0.5 * x0**2 + 1
            slope = x0 # Derivative of 0.5x^2 + 1 is x
            # Extend line segments symmetrically from the point
            p1 = axes.c2p(x0 - 0.9, y0 - 0.9 * slope)
            p2 = axes.c2p(x0 + 0.9, y0 + 0.9 * slope)
            line.set_points_as_corners([p1, p2])
        
        # Initial position
        update_tangent(tangent)
        tangent.add_updater(update_tangent)
        
        # Moving point on curve
        dot = Dot(color=TANGENT_COLOR)
        dot.add_updater(lambda d: d.move_to(axes.c2p(x_tracker.get_value(), 0.5 * x_tracker.get_value()**2 + 1)))
        
        self.play(FadeIn(dot), Create(tangent))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: It tells us how fast a value is changing.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(SLOPE_COLOR)
        )
        
        # Slope value label
        slope_label = MathTex(r"\text{Slope} = ", color=SLOPE_COLOR)
        slope_value = DecimalNumber(x_tracker.get_value(), num_decimal_places=2, color=SLOPE_COLOR)
        slope_group = VGroup(slope_label, slope_value).arrange(RIGHT, buff=0.2)
        self.place_in_area(slope_group, "A2", "A4", scale_factor=0.9)
        
        # Asset: speedometer.svg
        speedometer_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg"
        speedometer = SVGMobject(speedometer_asset, color=SLOPE_COLOR)
        self.place_at_grid(speedometer, "A5", scale_factor=0.7)
        
        # Needle for speedometer
        needle = Line(ORIGIN, UP * 0.3, color=RED, stroke_width=4)
        
        def update_needle(n):
            val = x_tracker.get_value() # slope range is approx -1.2 to 1.2
            # Map slope to angle. Let 0 be UP (PI/2).
            # Positive slope -> rotated right (clockwise, so negative increment from PI/2)
            angle = -val * (PI/3) 
            center = speedometer.get_center()
            # The speedometer SVG center is used as the pivot.
            n.set_points_as_corners([
                center,
                center + 0.35 * np.array([np.cos(PI/2 + angle), np.sin(PI/2 + angle), 0])
            ])

        needle.add_updater(update_needle)
        
        # Efficiently update decimal number value in place
        slope_value.add_updater(lambda v: v.set_value(x_tracker.get_value()))
        
        self.play(FadeIn(slope_group), FadeIn(speedometer), Create(needle))
        self.wait(0.5)
        
        # Animate the movement of the point, updating tangent and slope display
        self.play(x_tracker.animate.set_value(1.2), run_time=5, rate_func=rate_functions.smooth)
        self.wait(1)
        self.play(x_tracker.animate.set_value(-1.2), run_time=3, rate_func=rate_functions.smooth)
        self.wait(2)
