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
        # Setup layout with specific lecture lines
        lecture_lines_text = [
            "Coffee cools faster when it is much hotter.",
            "This cooling rate follows an e based curve.",
            "e models how systems return to equilibrium."
        ]
        self.setup_layout("Application: The Cooling Effect", lecture_lines_text)

        # === Animation for Lecture Line 1 ===
        # Coffee cools faster when it is much hotter.
        self.play(self.lecture[0].animate.set_color(RED), run_time=0.5)

        # Create Axes for the graph
        # Issue 54: Scale axes factor 0.7
        axes = Axes(
            x_range=[0, 4.5, 1],
            y_range=[0, 1.5, 0.5],
            axis_config={"include_tip": True, "color": WHITE},
            x_length=4,
            y_length=3
        )
        self.place_in_area(axes, "B2", "F6", scale_factor=0.7)
        
        labels = axes.get_axis_labels(
            x_label=Text("Time", font_size=20), 
            y_label=Text("Temp", font_size=20)
        )
        
        # Exponential decay curve y = e^-x
        curve = axes.plot(lambda x: np.exp(-x), x_range=[0, 4], color="#FF0000")
        
        # Issue 53: Place curve_label in A4-A5 area
        curve_label = Text("T(t) = e^-t", color="#FF0000", font_size=24)
        self.place_in_area(curve_label, "A4", "A5", scale_factor=0.8)

        # Asset Integration (Issue 37): Coffee cup
        coffee_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/coffee.svg")
        self.place_at_grid(coffee_icon, "A6", scale_factor=0.6)

        self.play(Create(axes), Write(labels), FadeIn(coffee_icon))
        self.play(Create(curve), Write(curve_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This cooling rate follows an e based curve.
        self.play(self.lecture[1].animate.set_color(YELLOW), run_time=0.5)

        # Setup ValueTracker for animation
        t_tracker = ValueTracker(0.2)

        # Tangent line (yellow)
        # Using a persistent object with an updater for efficiency
        tangent_line = Line(ORIGIN, RIGHT, color="#FFFF00", stroke_width=4)
        def update_tangent(line):
            t = t_tracker.get_value()
            slope = -np.exp(-t)
            point = axes.c2p(t, np.exp(-t))
            # Define line segment length 1.6 total (0.8 each side)
            line.set_points_as_corners([
                point + LEFT * 0.8 + UP * (0.8 * -slope),
                point + RIGHT * 0.8 + DOWN * (0.8 * -slope)
            ])
        tangent_line.add_updater(update_tangent)

        # Dot on the curve
        dot = Dot(color=YELLOW)
        dot.add_updater(lambda d: d.move_to(axes.c2p(t_tracker.get_value(), np.exp(-t_tracker.get_value()))))

        self.play(Create(tangent_line), FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # e models how systems return to equilibrium.
        self.play(self.lecture[2].animate.set_color(ORANGE), run_time=0.5)

        # 'Cooling Rate' label and value
        rate_label = Text("Cooling Rate:", font_size=20, color=WHITE)
        rate_value = DecimalNumber(
            np.exp(-t_tracker.get_value()),
            num_decimal_places=3,
            include_sign=False,
            font_size=20,
            color=YELLOW,
            mob_class=Text
        )
        rate_group = VGroup(rate_label, rate_value).arrange(RIGHT, buff=0.2)
        
        # Issue 52: Place rate_group in area B4-B5
        self.place_in_area(rate_group, 'B4', 'B5', scale_factor=0.8)

        # Update the value based on the tracker
        rate_value.add_updater(lambda d: d.set_value(np.exp(-t_tracker.get_value())))
        
        # Position following: offset from dot, but check constraints
        # To strictly follow Issue 52's spirit of reducing clutter, 
        # I'll update the group to follow the dot with a safe offset.
        rate_group.add_updater(lambda g: g.move_to(dot.get_center() + UP * 0.7 + RIGHT * 0.7))

        self.play(FadeIn(rate_group))
        
        # Animate the cooling process
        self.play(t_tracker.animate.set_value(3.5), run_time=5, rate_func=linear)
        self.wait(2)
