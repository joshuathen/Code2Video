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
        # Section Title and Lecture Lines
        title_str = "The Golden Rule: Sample Size (n ≥ 30)"
        lines_str = [
            "Small samples often result in messy distributions.",
            "Increasing the sample size fills the power meter.",
            "Reliability is reached once n reaches thirty."
        ]
        self.setup_layout(title_str, lines_str)
        
        # Define Colors
        RED_COLOR = "#FF0000"
        GREEN_COLOR = "#00FF00"
        YELLOW_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(RED_COLOR))

        # Visuals: Jagged red histogram
        # heights representing a "messy" distribution
        jagged_heights = [0.6, 1.4, 0.4, 1.1, 0.3, 1.5, 0.7, 1.2, 0.2, 0.8]
        bars_red = VGroup(*[
            Rectangle(
                width=0.2, 
                height=h, 
                fill_opacity=0.8, 
                fill_color=RED_COLOR, 
                stroke_color=WHITE, 
                stroke_width=1
            )
            for h in jagged_heights
        ]).arrange(RIGHT, buff=0.1)
        # Fix: Issue 46 - Move from B1-D4 to B2-D5
        self.place_in_area(bars_red, "B2", "D5")
        
        # Label: n = 2
        label_n = Text("n = 2", font_size=24, color=RED_COLOR)
        # Fix: Issue 47 - Move from A2 to A3
        self.place_at_grid(label_n, "A3")

        # Visuals: Power Meter Asset
        meter_svg = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/meter.svg", color=WHITE)
        # Fix: Issue 48 - Move from B5-E5 to B6-E6
        self.place_in_area(meter_svg, "B6", "E6", scale_factor=0.8)
        
        # Value Tracker for sample size n
        n_tracker = ValueTracker(2)
        
        # Meter fill (initial low red state)
        meter_fill = Rectangle(
            width=meter_svg.width * 0.6, 
            height=0.1, 
            fill_color=RED_COLOR, 
            fill_opacity=1, 
            stroke_width=0
        )
        meter_fill.move_to(meter_svg.get_bottom(), aligned_edge=DOWN).shift(UP * 0.1)

        # Updater for power meter fill bar
        def update_meter(m):
            val = n_tracker.get_value()
            # Linear mapping from n=[2, 30] to height proportional to meter_svg
            max_height = meter_svg.height * 0.8
            min_height = 0.1
            target_h = interpolate(min_height, max_height, (val - 2) / 28)
            m.stretch_to_fit_height(target_h)
            m.move_to(meter_svg.get_bottom(), aligned_edge=DOWN).shift(UP * 0.1)
            # Color transition logic
            if val < 30:
                m.set_color(RED_COLOR)
            else:
                m.set_color(GREEN_COLOR)

        meter_fill.add_updater(update_meter)

        self.play(
            Create(bars_red),
            Write(label_n),
            FadeIn(meter_svg),
            FadeIn(meter_fill)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(self.lecture[1].animate.set_color(GREEN_COLOR))

        # Target label for transformation
        label_n_30 = Text("n = 30", font_size=24, color=GREEN_COLOR)
        label_n_30.move_to(label_n.get_center())

        # Animate sample size increase and power meter fill
        self.play(
            n_tracker.animate.set_value(30),
            Transform(label_n, label_n_30),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(self.lecture[2].animate.set_color(YELLOW_COLOR))

        # Smooth Green Bell Curve (Normal Distribution)
        bell_curve = ParametricFunction(
            lambda t: np.array([t, 1.8 * np.exp(-t**2), 0]),
            t_range=[-2, 2],
            color=GREEN_COLOR,
            stroke_width=4
        )
        # Positioning bell curve within the same area as the histogram
        # Fix: Issue 46 - Match bars_red area B2-D5
        self.place_in_area(bell_curve, "B2", "D5")

        # Transform jagged histogram to smooth bell curve
        self.play(
            ReplacementTransform(bars_red, bell_curve),
            label_n.animate.set_color(YELLOW_COLOR)
        )

        # Pulse effect for the 'Magic Number' n = 30
        self.play(
            Indicate(label_n, color=YELLOW_COLOR, scale_factor=1.4),
            Flash(label_n, color=YELLOW_COLOR, line_length=0.3)
        )
        self.wait(2)
