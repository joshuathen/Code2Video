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
            "Let's move the second point closer to the first.",
            "The interval between them, h, starts to shrink.",
            "As h approaches zero, the secant line rotates.",
            "Zooming in, the curve begins to look straight.",
            "The two points eventually merge into one."
        ]
        self.setup_layout("The 'Zooming In' Technique (The Limit Concept)", lecture_lines)

        # Colors
        h_color = "#2ECC71"
        secant_color = "#E74C3C"
        point_a_color = "#3498DB"
        point_b_color = "#F1C40F"
        curve_color = "#95A5A6"

        # Math Params
        def f(x):
            return 0.15 * (x**2)
        
        # 1. Create Axes and Curve
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            axis_config={"include_tip": False, "color": GRAY},
            x_length=5,
            y_length=5
        )
        curve = axes.plot(f, x_range=[0, 5.5], color=curve_color)
        
        # 2. Points and Trackers
        x_a = 1.5
        x_b_init = 4.5
        h_tracker = ValueTracker(x_b_init - x_a)
        
        dot_a = Dot(axes.c2p(x_a, f(x_a)), color=point_a_color)
        dot_b = Dot(axes.c2p(x_a + h_tracker.get_value(), f(x_a + h_tracker.get_value())), color=point_b_color)
        
        # Update logic for dot_b
        dot_b.add_updater(lambda d: d.move_to(axes.c2p(x_a + h_tracker.get_value(), f(x_a + h_tracker.get_value()))))
        
        # 3. Secant Line with persistent updater
        secant_line = Line(color=secant_color, stroke_width=4)
        def update_secant(line):
            h = h_tracker.get_value()
            p1 = axes.c2p(x_a, f(x_a))
            if h < 0.01:
                # Limit slope (derivative)
                slope = 0.3 * x_a
                p2 = axes.c2p(x_a + 1, f(x_a) + slope)
            else:
                p2 = axes.c2p(x_a + h, f(x_a + h))
            
            direction = p2 - p1
            if np.linalg.norm(direction) > 0.001:
                direction = direction / np.linalg.norm(direction)
                line.set_points_as_corners([p1 - direction * 4, p1 + direction * 4])
            
        secant_line.add_updater(update_secant)

        # 4. h Indicator (Dynamic)
        h_line = Line(color=h_color, stroke_width=6)
        h_label_math = MathTex("h", color=h_color, font_size=32)
        
        def update_h_line(line):
            line.set_points_as_corners([
                axes.c2p(x_a, 0.4),
                axes.c2p(x_a + h_tracker.get_value(), 0.4)
            ])
            if h_tracker.get_value() < 0.05:
                line.set_opacity(0)
            else:
                line.set_opacity(1)

        def update_h_label(lbl):
            lbl.next_to(h_line, UP, buff=0.1)
            if h_tracker.get_value() < 0.2:
                lbl.set_opacity(0)
            else:
                lbl.set_opacity(1)

        h_line.add_updater(update_h_line)
        h_label_math.add_updater(update_h_label)
        
        # Static h-indicator icon for grid placement (Issue 28)
        h_icon_line = Line(LEFT, RIGHT, color=h_color).scale(0.3)
        h_icon_text = MathTex("h", color=h_color, font_size=24).next_to(h_icon_line, UP, buff=0.1)
        h_indicator_group = VGroup(h_icon_line, h_icon_text)

        # 5. Grouping and positioning
        graph_elements = VGroup(axes, curve, dot_a, dot_b, secant_line)
        
        # Apply Issue 26: Positioning the graph
        self.place_in_area(graph_elements, 'A2', 'E6', scale_factor=0.9)
        
        # Labels for Points (Issue 27)
        label_a = Text("A", font_size=24, color=point_a_color)
        label_b = Text("B", font_size=24, color=point_b_color)
        self.place_at_grid(label_a, 'E3', scale_factor=0.8)
        self.place_at_grid(label_b, 'B5', scale_factor=0.8)
        
        # Make label_b follow dot_b once animation starts
        label_b.add_updater(lambda l: l.next_to(dot_b, UP, buff=0.1))

        # Apply Issue 28: Alignment for h indicator
        self.place_at_grid(h_indicator_group, 'F3', scale_factor=0.7)

        # Initial Add
        self.add(axes, curve, dot_a, dot_b, secant_line, label_a, label_b)

        # === Animation for Lecture Line 1 ===
        # "Let's move the second point closer to the first."
        self.play(self.lecture[0].animate.set_color(point_b_color))
        self.play(h_tracker.animate.set_value(2.0), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The interval between them, h, starts to shrink."
        self.play(self.lecture[1].animate.set_color(h_color))
        self.add(h_line, h_label_math, h_indicator_group)
        self.play(h_tracker.animate.set_value(1.0), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "As h approaches zero, the secant line rotates."
        self.play(self.lecture[2].animate.set_color(secant_color))
        self.play(h_tracker.animate.set_value(0.4), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Zooming in, the curve begins to look straight."
        zoom_group = VGroup(axes, curve, dot_a, dot_b, secant_line, label_a, label_b, h_line, h_label_math)
        self.play(
            zoom_group.animate.scale(4, about_point=dot_a.get_center()),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The two points eventually merge into one."
        self.play(self.lecture[4].animate.set_color(point_a_color))
        self.play(
            h_tracker.animate.set_value(0.0001),
            label_b.animate.set_opacity(0),
            h_indicator_group.animate.set_opacity(0),
            run_time=3
        )
        self.wait(2)
